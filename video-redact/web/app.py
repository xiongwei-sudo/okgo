"""FastAPI：先把上传落到稳定路径，再提交单卡队列。页面同源，不启用 CORS。"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pipeline.enhance import EnhanceOptions
from pipeline.pipeline import RedactOptions, run_redact
from pipeline.queue import GPUJobQueue, JobState

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
WEIGHTS = ROOT / "weights"
JOBS.mkdir(exist_ok=True)

app = FastAPI(title="视频打码工具", version="0.2.0")
queue = GPUJobQueue(max_workers=1)

_SAFE_SUFFIX = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_MODES = {"gaussian", "mosaic", "solid"}

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>视频打码</title>
<style>
  body{font-family:sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
  label{display:block;margin:.35rem 0}
  fieldset{margin:1rem 0;border:1px solid #ddd;border-radius:8px}
  button{margin-top:.6rem;padding:.45rem 1rem}
  .bar{height:14px;background:#eee;border-radius:7px;overflow:hidden;margin:.5rem 0}
  .bar>div{height:100%;width:0;background:#2563eb}
  #status{font-weight:600}
  #err{color:#b00020;white-space:pre-wrap}
  #msg{color:#333;white-space:pre-wrap}
  .hidden{display:none}
</style>
</head><body>
<h1>视频打码工具</h1>
<p>上传后先保存到任务目录，再进入单卡队列。增强项在官方权重缺失时使用 OpenCV 降级。</p>
<form id="f">
  <label>视频 <input type="file" name="file" accept="video/*" required/></label>
  <fieldset><legend>打码</legend>
    <label><input type="checkbox" name="face" value="true" checked/> 人脸</label>
    <label><input type="checkbox" name="plate" value="true" checked/> 车牌</label>
    <label><input type="checkbox" name="text" value="true" checked/> 敏感文字</label>
    <label>模式
      <select name="mode">
        <option value="gaussian">高斯</option>
        <option value="mosaic">马赛克</option>
        <option value="solid">遮罩</option>
      </select>
    </label>
  </fieldset>
  <fieldset><legend>增强（无权重时自动降级）</legend>
    <label><input type="checkbox" name="stabilize" value="true"/> 稳像（vidstab / OpenCV）</label>
    <label><input type="checkbox" name="lowlight" value="true"/> 低光（gamma + CLAHE）</label>
    <label><input type="checkbox" name="denoise" value="true"/> 去噪（fastNlMeans）</label>
    <label><input type="checkbox" name="deblur" value="true"/> 去糊（反锐化）</label>
    <label><input type="checkbox" name="superres" value="true"/> 超分（三次插值）</label>
    <label>超分倍数
      <select name="scale"><option value="2">2</option><option value="4">4</option></select>
    </label>
  </fieldset>
  <button id="go" type="submit">开始处理</button>
</form>
<section id="panel" class="hidden">
  <p>任务 <code id="jid"></code></p>
  <p id="status">排队中</p>
  <div class="bar" aria-hidden="true"><div id="bar"></div></div>
  <p id="pct">0%</p>
  <p id="msg"></p>
  <p id="err"></p>
  <a id="dl" class="hidden" href="#">下载结果</a>
</section>
<script>
const STATUS = {queued:"排队中", running:"处理中", done:"已完成", failed:"失败"};
const log = {
  panel: document.getElementById("panel"),
  status: document.getElementById("status"),
  bar: document.getElementById("bar"),
  pct: document.getElementById("pct"),
  msg: document.getElementById("msg"),
  err: document.getElementById("err"),
  dl: document.getElementById("dl"),
  jid: document.getElementById("jid"),
  go: document.getElementById("go"),
};
function show(job) {
  log.panel.classList.remove("hidden");
  const pct = Math.max(0, Math.min(100, Math.round((job.progress || 0) * 100)));
  log.status.textContent = STATUS[job.status] || job.status || "处理中";
  log.bar.style.width = pct + "%";
  log.pct.textContent = pct + "%";
  log.msg.textContent = job.message || "";
  log.err.textContent = job.error || "";
  if (job.status === "done") {
    log.dl.href = "/api/jobs/" + job.job_id + "/download";
    log.dl.textContent = "下载结果";
    log.dl.classList.remove("hidden");
  } else {
    log.dl.classList.add("hidden");
  }
}
document.getElementById("f").onsubmit = async (e) => {
  e.preventDefault();
  log.go.disabled = true;
  log.dl.classList.add("hidden");
  log.err.textContent = "";
  log.msg.textContent = "正在保存上传文件…";
  log.panel.classList.remove("hidden");
  log.status.textContent = "上传中";
  try {
    const r = await fetch("/api/jobs", {method:"POST", body:new FormData(e.target)});
    const j = await r.json();
    if (!r.ok) {
      log.status.textContent = "未开始";
      log.err.textContent = j.error || JSON.stringify(j);
      return;
    }
    log.jid.textContent = j.job_id;
    const poll = async () => {
      const s = await (await fetch("/api/jobs/" + j.job_id)).json();
      s.job_id = j.job_id;
      show(s);
      if (s.status === "done" || s.status === "failed") return;
      setTimeout(poll, 800);
    };
    poll();
  } catch (err) {
    log.err.textContent = String(err);
  } finally {
    log.go.disabled = false;
  }
};
</script>
</body></html>
"""


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in _SAFE_SUFFIX else ".mp4"


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    face: str = Form("false"),
    plate: str = Form("false"),
    text: str = Form("false"),
    mode: str = Form("gaussian"),
    stabilize: str = Form("false"),
    lowlight: str = Form("false"),
    denoise: str = Form("false"),
    deblur: str = Form("false"),
    superres: str = Form("false"),
    scale: str = Form("2"),
):
    use_face = _flag(face)
    use_plate = _flag(plate)
    use_text = _flag(text)
    use_stabilize = _flag(stabilize)
    use_lowlight = _flag(lowlight)
    use_denoise = _flag(denoise)
    use_deblur = _flag(deblur)
    use_superres = _flag(superres)
    if mode not in _MODES:
        return JSONResponse({"error": "不支持的打码模式"}, status_code=400)
    if not (use_face or use_plate or use_text):
        return JSONResponse({"error": "至少选择人脸、车牌或敏感文字之一"}, status_code=400)
    try:
        scale_i = int(scale)
    except ValueError:
        scale_i = 0
    if scale_i not in (2, 4):
        return JSONResponse({"error": "超分倍数只能是 2 或 4"}, status_code=400)

    # 先落盘到 jobs/<id>/input.*，再入队。路径只由任务号决定，避免共用文件名互相覆盖。
    jid = uuid.uuid4().hex[:12]
    jdir = JOBS / jid
    jdir.mkdir(parents=True, exist_ok=True)
    inp = jdir / f"input{_suffix(file.filename)}"
    out = jdir / "output.mp4"
    written = 0
    try:
        with inp.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
        await file.close()
    except Exception:
        logger.exception("save upload failed")
        shutil.rmtree(jdir, ignore_errors=True)
        return JSONResponse({"error": "保存上传文件失败"}, status_code=500)
    if written < 1 or not inp.is_file():
        shutil.rmtree(jdir, ignore_errors=True)
        return JSONResponse({"error": "空文件"}, status_code=400)

    enhance = EnhanceOptions(
        enable=any((use_stabilize, use_lowlight, use_denoise, use_deblur, use_superres)),
        stabilize=use_stabilize,
        lowlight=use_lowlight,
        denoise=use_denoise,
        deblur=use_deblur,
        superres=use_superres,
        scale=scale_i,
    )
    opts = RedactOptions(
        face=use_face,
        plate=use_plate,
        text=use_text,
        mode=mode,
        enhance=enhance,
    )
    logger.info("job %s saved %s (%d bytes) before submit", jid, inp, written)

    def _run(state: JobState) -> None:
        state.message = "已入队，开始处理"

        def prog(frac: float, msg: str) -> None:
            state.progress = frac
            state.message = msg

        result = run_redact(inp, out, WEIGHTS, opts, progress=prog)
        state.output = result.output
        state.message = "; ".join(result.notes) if result.notes else "完成"

    queue.submit(_run, job_id=jid)
    return {"job_id": jid}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    st = queue.get(job_id)
    if not st:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "job_id": st.job_id,
        "status": st.status.value,
        "progress": st.progress,
        "message": st.message,
        "error": st.error,
    }


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    st = queue.get(job_id)
    if not st or not st.output or not Path(st.output).exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(st.output, filename=f"{job_id}.mp4", media_type="video/mp4")
