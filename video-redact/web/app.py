"""FastAPI：上传 / 进度 / 下载；支持增强开关。"""
from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pipeline.config import JOBS_DIR, WEIGHTS_DIR
from pipeline.enhance import EnhanceOptions
from pipeline.pipeline import RedactOptions, run_redact
from pipeline.queue import GPUJobQueue, JobStatus

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="视频打码工具", version="0.1.1")
queue = GPUJobQueue(max_workers=1)

_SAFE_EXT = re.compile(r"^[a-z0-9]{1,8}$", re.I)

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>视频打码</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.45}
label{display:block;margin:.45rem 0}
fieldset{border:1px solid #ccc;border-radius:8px;margin:1rem 0;padding:.75rem 1rem}
legend{padding:0 .4rem;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:1rem}
.bar{height:10px;background:#eee;border-radius:6px;overflow:hidden;margin:.5rem 0}
.bar>i{display:block;height:100%;width:0;background:#2b7cff;transition:width .2s}
#status{font-weight:600}
#log{background:#f6f8fa;padding:.75rem;border-radius:8px;white-space:pre-wrap;word-break:break-all}
a#dl{display:none;margin-top:.5rem}
</style>
</head><body>
<h1>视频打码工具</h1>
<form id="f">
<label>视频 <input type="file" name="file" accept="video/*" required/></label>
<fieldset><legend>打码目标</legend>
<label><input type="checkbox" name="face" checked/> 人脸</label>
<label><input type="checkbox" name="plate" checked/> 车牌</label>
<label><input type="checkbox" name="text" checked/> 敏感文字</label>
<label>模式
<select name="mode"><option value="gaussian">高斯</option><option value="mosaic">马赛克</option><option value="solid">遮罩</option></select>
</label>
</fieldset>
<fieldset><legend>增强（缺权重时 OpenCV 降级）</legend>
<label><input type="checkbox" name="enhance"/> 启用增强</label>
<div class="row">
<label><input type="checkbox" name="stabilize"/> 稳像 N3</label>
<label><input type="checkbox" name="lowlight"/> 低光 N2</label>
<label><input type="checkbox" name="denoise"/> 去噪 N2</label>
<label><input type="checkbox" name="deblur"/> 去糊 N2</label>
<label><input type="checkbox" name="superres"/> 超分 N1</label>
</div>
<label>超分倍率
<select name="scale"><option value="2">2x</option><option value="4">4x</option></select>
</label>
</fieldset>
<button type="submit">开始处理</button>
</form>
<p id="status">空闲</p>
<div class="bar"><i id="prog"></i></div>
<pre id="log"></pre>
<a id="dl">下载结果</a>
<script>
const log=document.getElementById('log');
const dl=document.getElementById('dl');
const statusEl=document.getElementById('status');
const prog=document.getElementById('prog');
const BOOLS=['face','plate','text','enhance','stabilize','lowlight','denoise','deblur','superres'];
function setProgress(p, label){
  const pct=Math.max(0,Math.min(100, Math.round((p||0)*100)));
  prog.style.width=pct+'%';
  statusEl.textContent=label || (pct+'%');
}
document.getElementById('f').onsubmit=async(e)=>{
  e.preventDefault();
  dl.style.display='none';
  const fd=new FormData(e.target);
  BOOLS.forEach(k=>{ if(!fd.has(k)) fd.set(k,'false'); else fd.set(k,'true'); });
  // 勾选任一增强子项时自动 enable
  if(['stabilize','lowlight','denoise','deblur','superres'].some(k=>fd.get(k)==='true')){
    fd.set('enhance','true');
  }
  setProgress(0,'上传中…');
  log.textContent='正在上传并创建任务…';
  let r,j;
  try{
    r=await fetch('/api/jobs',{method:'POST',body:fd});
    j=await r.json();
  }catch(err){
    setProgress(0,'上传失败');
    log.textContent=String(err);
    return;
  }
  if(!r.ok){setProgress(0,'创建失败');log.textContent=JSON.stringify(j,null,2);return;}
  const id=j.job_id;
  setProgress(0,'已入队 '+id);
  const poll=async()=>{
    try{
      const s=await (await fetch('/api/jobs/'+id)).json();
      const label=(s.stage||s.status||'')+(s.message?(' · '+s.message):'');
      setProgress(s.progress||0, label);
      log.textContent=JSON.stringify(s,null,2);
      if(s.status==='done'){
        setProgress(1,'完成');
        dl.href='/api/jobs/'+id+'/download';
        dl.style.display='inline';
        dl.textContent='下载结果 '+id+'.mp4';
        return;
      }
      if(s.status==='failed'){
        setProgress(s.progress||0,'失败');
        return;
      }
      setTimeout(poll,800);
    }catch(err){
      log.textContent=String(err);
      setTimeout(poll,1500);
    }
  };
  poll();
};
</script></body></html>
"""


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def _safe_ext(filename: str | None) -> str:
    if not filename:
        return "mp4"
    name = Path(filename).name  # drop any path segments
    suf = Path(name).suffix.lstrip(".").lower()
    if suf and _SAFE_EXT.match(suf):
        return suf
    return "mp4"


def _status_payload(st) -> dict:
    """统一状态字段，便于前端展示。"""
    stage_map = {
        JobStatus.queued: "排队中",
        JobStatus.running: "处理中",
        JobStatus.done: "完成",
        JobStatus.failed: "失败",
    }
    stage = stage_map.get(st.status, st.status.value)
    pct = round(float(st.progress or 0.0) * 100.0, 1)
    msg = st.message or ""
    if st.status == JobStatus.failed and st.error:
        msg = st.error
    return {
        "job_id": st.job_id,
        "status": st.status.value,
        "stage": stage,
        "progress": float(st.progress or 0.0),
        "progress_pct": pct,
        "message": msg,
        "error": st.error,
        "has_output": bool(st.output and Path(st.output).exists()),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    face: str = Form("true"),
    plate: str = Form("true"),
    text: str = Form("true"),
    mode: str = Form("gaussian"),
    enhance: str = Form("false"),
    stabilize: str = Form("false"),
    lowlight: str = Form("false"),
    denoise: str = Form("false"),
    deblur: str = Form("false"),
    superres: str = Form("false"),
    scale: str = Form("2"),
):
    """先落盘到临时文件再原子移入 jobs/<id>/，避免半写文件被 worker 读到。"""
    jid = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / jid
    jdir.mkdir(parents=True, exist_ok=True)

    ext = _safe_ext(file.filename)
    final_inp = jdir / f"input.{ext}"

    # 上传到同目录临时文件，写完再 replace，消除读半文件竞态
    fd, tmp_name = tempfile.mkstemp(prefix=f".up_{jid}_", suffix=f".{ext}", dir=str(jdir))
    tmp_path = Path(tmp_name)
    try:
        import os

        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)
            out.flush()
            os.fsync(out.fileno())
        tmp_path.replace(final_inp)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    face_b = _as_bool(face)
    plate_b = _as_bool(plate)
    text_b = _as_bool(text)
    stabilize_b = _as_bool(stabilize)
    lowlight_b = _as_bool(lowlight)
    denoise_b = _as_bool(denoise)
    deblur_b = _as_bool(deblur)
    superres_b = _as_bool(superres)
    enhance_b = _as_bool(enhance) or any(
        [stabilize_b, lowlight_b, denoise_b, deblur_b, superres_b]
    )
    try:
        scale_i = int(scale)
    except ValueError:
        scale_i = 2
    if scale_i not in (2, 4):
        scale_i = 2
    if mode not in ("gaussian", "mosaic", "solid"):
        mode = "gaussian"

    def _run(state) -> None:
        state.message = "准备流水线"
        state.progress = 0.01
        out = jdir / "output.mp4"
        enh = EnhanceOptions(
            enable=enhance_b,
            stabilize=stabilize_b,
            lowlight=lowlight_b,
            denoise=denoise_b,
            deblur=deblur_b,
            superres=superres_b,
            scale=scale_i,
        )
        opts = RedactOptions(
            face=face_b,
            plate=plate_b,
            text=text_b,
            mode=mode,
            enhance=enh,
        )

        def prog(p: float, msg: str) -> None:
            state.progress = max(0.01, min(0.99, float(p)))
            state.message = msg

        result = run_redact(final_inp, out, WEIGHTS_DIR, opts, progress=prog)
        state.output = result.output
        note = "; ".join(result.notes) if result.notes else "处理完成"
        state.message = f"{note} · frames={result.frames} det={result.detections_total}"

    queue.submit(_run, job_id=jid)
    return {
        "job_id": jid,
        "status": "queued",
        "stage": "排队中",
        "message": "已创建，等待 GPU 任务",
        "input": final_inp.name,
        "enhance": enhance_b,
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    st = queue.get(job_id)
    if not st:
        return JSONResponse({"error": "not found", "job_id": job_id}, status_code=404)
    return _status_payload(st)


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    st = queue.get(job_id)
    if not st or st.status != JobStatus.done or not st.output or not Path(st.output).exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(st.output, filename=f"{job_id}.mp4", media_type="video/mp4")
