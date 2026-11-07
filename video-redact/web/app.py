"""FastAPI 骨架：上传 / 进度 / 下载。一期可先用 CLI。"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pipeline.enhance import EnhanceOptions
from pipeline.pipeline import RedactOptions, run_redact
from pipeline.queue import GPUJobQueue, JobState

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
WEIGHTS = ROOT / "weights"
JOBS.mkdir(exist_ok=True)

app = FastAPI(title="视频打码工具", version="0.1.0")
queue = GPUJobQueue(max_workers=1)

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>视频打码</title>
<style>body{font-family:sans-serif;max-width:640px;margin:2rem auto}label{display:block;margin:.4rem 0}</style>
</head><body>
<h1>视频打码工具</h1>
<form id="f">
<label>视频 <input type="file" name="file" accept="video/*" required/></label>
<label><input type="checkbox" name="face" checked/> 人脸</label>
<label><input type="checkbox" name="plate" checked/> 车牌</label>
<label><input type="checkbox" name="text" checked/> 敏感文字</label>
<label>模式
<select name="mode"><option value="gaussian">高斯</option><option value="mosaic">马赛克</option><option value="solid">遮罩</option></select>
</label>
<button type="submit">开始处理</button>
</form>
<pre id="log"></pre>
<a id="dl" style="display:none">下载结果</a>
<script>
const log=document.getElementById('log');
const dl=document.getElementById('dl');
document.getElementById('f').onsubmit=async(e)=>{
  e.preventDefault();
  const fd=new FormData(e.target);
  log.textContent='上传中...';
  const r=await fetch('/api/jobs',{method:'POST',body:fd});
  const j=await r.json();
  if(!r.ok){log.textContent=JSON.stringify(j);return;}
  const id=j.job_id;
  const poll=async()=>{
    const s=await (await fetch('/api/jobs/'+id)).json();
    log.textContent=JSON.stringify(s,null,2);
    if(s.status==='done'){dl.href='/api/jobs/'+id+'/download';dl.style.display='inline';dl.textContent='下载结果';return;}
    if(s.status==='failed')return;
    setTimeout(poll,1000);
  };
  poll();
};
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    face: bool = Form(True),
    plate: bool = Form(True),
    text: bool = Form(True),
    mode: str = Form("gaussian"),
):
    job_dir = JOBS / "pending"
    job_dir.mkdir(parents=True, exist_ok=True)

    def _run(state: JobState) -> None:
        jdir = JOBS / state.job_id
        jdir.mkdir(parents=True, exist_ok=True)
        inp = jdir / (file.filename or "input.mp4")
        # file already saved below before submit — re-read path from state.message
        src = Path(state.message)
        shutil.move(str(src), str(inp))
        out = jdir / "output.mp4"
        opts = RedactOptions(
            face=face, plate=plate, text=text, mode=mode, enhance=EnhanceOptions(enable=False)
        )

        def prog(p: float, msg: str) -> None:
            state.progress = p
            state.message = msg

        result = run_redact(inp, out, WEIGHTS, opts, progress=prog)
        state.output = result.output
        state.message = "; ".join(result.notes) if result.notes else "done"

    # save upload first
    tmp = JOBS / "pending" / (file.filename or "upload.mp4")
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def _bound(state: JobState) -> None:
        state.message = str(tmp)
        _run(state)

    jid = queue.submit(_bound)
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
