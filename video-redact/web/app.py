"""FastAPI：上传 / 进度 / 下载；支持增强开关与任务取消／暂停／继续／重启。"""
from __future__ import annotations

import json
import time
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pipeline.config import JOBS_DIR, WEIGHTS_DIR
from pipeline.enhance import EnhanceOptions
from pipeline.pipeline import RedactOptions, run_redact
from pipeline.queue import AdmitDecision, GPUJobQueue, JobCancelled, JobStatus

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="视频打码工具", version="0.1.6")
queue = GPUJobQueue(max_workers=2, min_free_vram_mb_per_job=8000.0)

_SAFE_EXT = re.compile(r"^[a-z0-9]{1,8}$", re.I)

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>视频打码</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;line-height:1.45}
label{display:block;margin:.45rem 0}
fieldset{border:1px solid #ccc;border-radius:8px;margin:1rem 0;padding:.75rem 1rem}
legend{padding:0 .4rem;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:1rem}
.bar{height:10px;background:#eee;border-radius:6px;overflow:hidden;margin:.5rem 0}
.bar>i{display:block;height:100%;width:0;background:#2b7cff;transition:width .2s}
#status{font-weight:600}
#worker,#paths{font-size:.9rem;color:#333;margin:.25rem 0}
#log{background:#f6f8fa;padding:.75rem;border-radius:8px;white-space:pre-wrap;word-break:break-all;font-size:12px}
a#dl{display:none;margin-top:.5rem}
.ctrl{display:flex;flex-wrap:wrap;gap:.5rem;margin:.75rem 0}
.ctrl button{padding:.4rem .85rem;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
.ctrl button:disabled{opacity:.45;cursor:not-allowed}
.ctrl button.danger{border-color:#c44;color:#c44}
.hint{color:#666;font-size:.9rem;margin:.25rem 0 1rem}
.tag{display:inline-block;padding:.1rem .45rem;border-radius:4px;background:#eef;margin:.1rem .2rem 0 0;font-size:.8rem}
.tag.warn{background:#ffe9c6}
.tag.ok{background:#d9f5d9}
</style>
</head><body>
<h1>视频打码 / 增强 <small style="font-weight:400;color:#888">v0.1.6</small></h1>
<p class="hint">只开增强、关掉全部打码目标 → <strong>不会</strong>做人脸/车牌/OCR 马赛克。增强路径（官方 GPU / OpenCV 降级）会显示在下方与任务日志。</p>
<p id="worker">worker: …</p>
<p id="paths">enhance_path: （提交后可见）</p>
<form id="f">
<label>视频 <input type="file" name="file" accept="video/*" required/></label>
<fieldset><legend>打码目标（可全关 = 仅增强）</legend>
<label><input type="checkbox" name="face" id="face"/> 人脸</label>
<label><input type="checkbox" name="plate" id="plate"/> 车牌</label>
<label><input type="checkbox" name="text" id="text"/> 敏感文字</label>
<label>模式
<select name="mode"><option value="gaussian">高斯</option><option value="mosaic">马赛克</option><option value="solid">遮罩</option></select>
</label>
</fieldset>
<fieldset><legend>增强</legend>
<label><input type="checkbox" name="enhance" id="enhance"/> 启用增强</label>
<div class="row">
<label><input type="checkbox" name="stabilize" id="stabilize"/> 稳像 N3</label>
<label><input type="checkbox" name="lowlight" id="lowlight"/> 低光 Zero-DCE</label>
<label><input type="checkbox" name="denoise" id="denoise"/> 去噪 N2</label>
<label><input type="checkbox" name="deblur" id="deblur"/> 去糊 N2</label>
<label><input type="checkbox" name="superres" id="superres"/> 超分 Real-ESRGAN</label>
</div>
<label>超分倍率
<select name="scale"><option value="2">2x</option><option value="4">4x</option></select>
</label>
</fieldset>
<button type="submit">开始处理</button>
</form>
<p id="status">空闲</p>
<div class="bar"><i id="prog"></i></div>
<div class="ctrl">
<button type="button" id="btnPause" disabled>暂停</button>
<button type="button" id="btnResume" disabled>继续</button>
<button type="button" id="btnCancel" class="danger" disabled>取消</button>
<button type="button" id="btnRestart" disabled>重启</button>
</div>
<pre id="log"></pre>
<a id="dl">下载结果</a>
<script>
const log=document.getElementById('log');
const dl=document.getElementById('dl');
const statusEl=document.getElementById('status');
const prog=document.getElementById('prog');
const workerEl=document.getElementById('worker');
const pathsEl=document.getElementById('paths');
const btnPause=document.getElementById('btnPause');
const btnResume=document.getElementById('btnResume');
const btnCancel=document.getElementById('btnCancel');
const btnRestart=document.getElementById('btnRestart');
const BOOLS=['face','plate','text','enhance','stabilize','lowlight','denoise','deblur','superres'];
let currentId=null;
let polling=false;

function renderPaths(pr){
  if(!pr || typeof pr!=='object'){ pathsEl.textContent='enhance_path: （无）'; return; }
  const keys=Object.keys(pr);
  if(!keys.length){ pathsEl.textContent='enhance_path: （未启用）'; return; }
  pathsEl.innerHTML='enhance_path: '+keys.map(k=>{
    const v=String(pr[k]);
    const ok=v.startsWith('official');
    return '<span class="tag '+(ok?'ok':'warn')+'">'+k+': '+v+'</span>';
  }).join(' ');
}

async function refreshWorker(){
  try{
    const w=await (await fetch('/api/worker')).json();
    const g=w.gpu||{};
    const gpuBits=[g.available?('GPU '+(g.name||'ok')):'无GPU', g.idle?'空闲':'忙碌'];
    if(g.memory_used_mb!=null && g.memory_total_mb!=null){
      gpuBits.push(Math.round(g.memory_used_mb)+'/'+Math.round(g.memory_total_mb)+'MB');
    }
    workerEl.textContent='worker: '+(w.liveness||'?')
      +' · conc '+(w.concurrency||1)
      +' · slot '+(w.slot_free?'free':'busy')
      +' · active '+(w.active_jobs||0)+' queued '+(w.queued_jobs||0)
      +' · '+gpuBits.join(' · ')
      +(w.current_job_id?(' · job '+w.current_job_id):'');
    workerEl.style.color=(w.liveness==='hung'||w.liveness==='dead')?'#c44':'#333';
  }catch(err){
    workerEl.textContent='worker: 状态获取失败 '+err;
  }
}
refreshWorker();
setInterval(refreshWorker, 2500);

function setProgress(p, label){
  const pct=Math.max(0,Math.min(100, Math.round((p||0)*100)));
  prog.style.width=pct+'%';
  statusEl.textContent=label || (pct+'%');
}
function updateButtons(s){
  const st=(s && s.status) || '';
  const active=['queued','running','paused'].includes(st);
  btnCancel.disabled=!active;
  btnPause.disabled=!(st==='queued'||st==='running');
  btnResume.disabled=!(st==='paused');
  btnRestart.disabled=!(st==='done'||st==='failed'||st==='cancelled');
}
async function postAction(action){
  if(!currentId) return;
  try{
    const r=await fetch('/api/jobs/'+currentId+'/'+action,{method:'POST'});
    const j=await r.json();
    log.textContent=JSON.stringify(j,null,2);
    if(j.enhance_paths) renderPaths(j.enhance_paths);
    if(j.job_id && action==='restart'){
      currentId=j.job_id;
      dl.style.display='none';
      setProgress(0,'已重新入队 '+currentId);
      if(!polling) poll();
    }else if(j.status){
      updateButtons(j);
      setProgress(j.progress||0, (j.stage||j.status)+(j.message?(' · '+j.message):''));
    }
  }catch(err){
    log.textContent=String(err);
  }
}
btnPause.onclick=()=>postAction('pause');
btnResume.onclick=()=>postAction('resume');
btnCancel.onclick=()=>postAction('cancel');
btnRestart.onclick=()=>postAction('restart');

const poll=async()=>{
  if(!currentId){polling=false;return;}
  polling=true;
  try{
    const s=await (await fetch('/api/jobs/'+currentId)).json();
    const label=(s.stage||s.status||'')+(s.message?(' · '+s.message):'');
    setProgress(s.progress||0, label);
    log.textContent=JSON.stringify(s,null,2);
    if(s.enhance_paths) renderPaths(s.enhance_paths);
    updateButtons(s);
    if(s.status==='done'){
      setProgress(1,'完成');
      dl.href='/api/jobs/'+currentId+'/download';
      dl.style.display='inline';
      dl.textContent='下载结果 '+currentId+'.mp4';
      polling=false;
      return;
    }
    if(s.status==='failed'||s.status==='cancelled'){
      const why=s.fail_reason||s.error||s.message||'';
      setProgress(s.progress||0, s.status==='cancelled'?'已取消':('失败'+(why?(' · '+why):'')));
      polling=false;
      refreshWorker();
      return;
    }
    setTimeout(poll, s.status==='paused' ? 1200 : 800);
  }catch(err){
    log.textContent=String(err);
    setTimeout(poll,1500);
  }
};

document.getElementById('f').onsubmit=async(e)=>{
  e.preventDefault();
  dl.style.display='none';
  const fd=new FormData(e.target);
  BOOLS.forEach(k=>{ if(!fd.has(k)) fd.set(k,'false'); else fd.set(k,'true'); });
  if(['stabilize','lowlight','denoise','deblur','superres'].some(k=>fd.get(k)==='true')){
    fd.set('enhance','true');
  }
  const anyRedact=['face','plate','text'].some(k=>fd.get(k)==='true');
  const anyEnh=fd.get('enhance')==='true';
  if(!anyRedact && !anyEnh){
    setProgress(0,'请至少勾选一项打码目标，或启用增强');
    return;
  }
  setProgress(0,'上传中…');
  log.textContent='正在上传并创建任务…';
  updateButtons({});
  let r,j;
  try{
    r=await fetch('/api/jobs',{method:'POST',body:fd});
    j=await r.json();
  }catch(err){
    setProgress(0,'上传失败');
    log.textContent=String(err);
    return;
  }
  if(!r.ok){
    const why=(j && (j.fail_reason||j.message||j.error)) || ('HTTP '+r.status);
    setProgress(0,'无法启动 · '+why);
    log.textContent=JSON.stringify(j,null,2);
    refreshWorker();
    return;
  }
  currentId=j.job_id;
  const admit=j.admit||'queued';
  const modeHint=(!anyRedact && anyEnh)?'〔仅增强〕 ':'';
  setProgress(0, modeHint+(admit==='run'?'立即开跑 ':'已入队 ')+currentId+(j.admit_reason?(' · '+j.admit_reason):''));
  updateButtons({status:j.status||'queued'});
  refreshWorker();
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
    name = Path(filename).name
    suf = Path(name).suffix.lstrip(".").lower()
    if suf and _SAFE_EXT.match(suf):
        return suf
    return "mp4"


def _options_path(jdir: Path) -> Path:
    return jdir / "options.json"


def _save_options(jdir: Path, opts: Dict[str, Any], input_name: str) -> None:
    payload = dict(opts)
    payload["input_name"] = input_name
    _options_path(jdir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_options(jdir: Path) -> Optional[Dict[str, Any]]:
    p = _options_path(jdir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_input(jdir: Path, preferred: Optional[str] = None) -> Optional[Path]:
    if preferred:
        cand = jdir / preferred
        if cand.exists():
            return cand
    matches = sorted(jdir.glob("input.*"))
    return matches[0] if matches else None


def _status_payload(st) -> dict:
    stage_map = {
        JobStatus.queued: "排队中",
        JobStatus.running: "处理中",
        JobStatus.paused: "已暂停",
        JobStatus.cancelled: "已取消",
        JobStatus.done: "完成",
        JobStatus.failed: "失败",
    }
    stage = stage_map.get(st.status, st.status.value)
    pct = round(float(st.progress or 0.0) * 100.0, 1)
    msg = st.message or ""
    if st.status == JobStatus.failed and st.error:
        msg = st.error
    snap = queue.worker_snapshot()
    return {
        "job_id": st.job_id,
        "status": st.status.value,
        "stage": stage,
        "progress": float(st.progress or 0.0),
        "progress_pct": pct,
        "message": msg,
        "error": st.error,
        "fail_reason": getattr(st, "fail_reason", None) or (st.error if st.status == JobStatus.failed else None),
        "admit": getattr(st, "admit", None),
        "admit_reason": getattr(st, "admit_reason", None),
        "wants_enhance": bool(getattr(st, "wants_enhance", False)),
        "process": {
            "heartbeat_age_s": (
                round(time.time() - st.heartbeat_at, 2) if getattr(st, "heartbeat_at", None) else None
            ),
            "worker_tid": getattr(st, "worker_tid", None),
            "started_at": getattr(st, "started_at", None),
            "future_alive": bool(
                st.job_id in getattr(queue, "_futures", {})
                and not queue._futures[st.job_id].done()
            ) if st.job_id in getattr(queue, "_futures", {}) else False,
        },
        "worker": {
            "liveness": snap.get("liveness"),
            "slot_free": snap.get("slot_free"),
            "current_job_id": snap.get("current_job_id"),
            "queued_jobs": snap.get("queued_jobs"),
        },
        "has_output": bool(st.output and Path(st.output).exists()),
        "enhance_paths": getattr(st, "enhance_paths", None) or {},
        "controls": {
            "can_cancel": st.status in (JobStatus.queued, JobStatus.running, JobStatus.paused),
            "can_pause": st.status in (JobStatus.queued, JobStatus.running),
            "can_resume": st.status == JobStatus.paused,
            "can_restart": st.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled),
            "pause_note": "暂停在下一帧安全点生效",
        },
    }


def _parse_options_from_form(
    face, plate, text, mode, enhance, stabilize, lowlight, denoise, deblur, superres, scale
) -> Dict[str, Any]:
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
    return {
        "face": face_b,
        "plate": plate_b,
        "text": text_b,
        "mode": mode,
        "enhance": enhance_b,
        "stabilize": stabilize_b,
        "lowlight": lowlight_b,
        "denoise": denoise_b,
        "deblur": deblur_b,
        "superres": superres_b,
        "scale": scale_i,
    }


def _submit_redact_job(
    jid: str,
    jdir: Path,
    inp: Path,
    opt: Dict[str, Any],
    *,
    admit: AdmitDecision,
    admit_reason: str,
) -> None:
    out = jdir / "output.mp4"
    wants_enhance = bool(opt.get("enhance"))

    def _run(state) -> None:
        queue.heartbeat(state)
        state.message = "准备流水线"
        state.progress = 0.01
        enh = EnhanceOptions(
            enable=bool(opt.get("enhance")),
            stabilize=bool(opt.get("stabilize")),
            lowlight=bool(opt.get("lowlight")),
            denoise=bool(opt.get("denoise")),
            deblur=bool(opt.get("deblur")),
            superres=bool(opt.get("superres")),
            scale=int(opt.get("scale") or 2),
        )
        opts = RedactOptions(
            face=bool(opt.get("face", False)),
            plate=bool(opt.get("plate", False)),
            text=bool(opt.get("text", False)),
            mode=str(opt.get("mode") or "gaussian"),
            enhance=enh,
        )

        def prog(p: float, msg: str) -> None:
            if state.cancel_flag.is_set():
                raise JobCancelled("cancelled")
            queue.heartbeat(state)
            if isinstance(msg, str) and msg.startswith("enhance_path:"):
                # "enhance_path: lowlight=..., superres=..."
                pr = {}
                body = msg.split(":", 1)[-1].strip()
                if body and body != "none":
                    for part in body.split(","):
                        part = part.strip()
                        if "=" in part:
                            k, v = part.split("=", 1)
                            pr[k.strip()] = v.strip()
                if pr:
                    state.enhance_paths = pr
            # 保持 paused 展示
            if state.status != JobStatus.paused:
                state.progress = max(0.01, min(0.99, float(p)))
                state.message = msg
            else:
                state.progress = max(0.01, min(0.99, float(p)))

        def control() -> None:
            queue.wait_if_paused(state)

        result = run_redact(
            inp, out, WEIGHTS_DIR, opts, progress=prog, control=control
        )
        queue.heartbeat(state)
        state.output = result.output
        state.enhance_paths = dict(result.enhance_paths or {})
        # 任务一开始加载完也会写 notes；中途把 path 挂到 message 前缀便于 UI
        if result.enhance_paths:
            state.enhance_paths = dict(result.enhance_paths)
        note = "; ".join(result.notes) if result.notes else "处理完成"
        state.message = f"{note} · frames={result.frames} det={result.detections_total}"

    queue.submit(
        _run,
        job_id=jid,
        wants_enhance=wants_enhance,
        admit=admit,
        admit_reason=admit_reason,
    )


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

    opt = _parse_options_from_form(
        face, plate, text, mode, enhance, stabilize, lowlight, denoise, deblur, superres, scale
    )
    # 调度判断：worker 存活 / GPU 空闲 / 槽位 → run|queue|reject
    queue.reconcile()
    decision, reason = queue.admit(wants_enhance=bool(opt["enhance"]))
    if decision == AdmitDecision.reject:
        # 清理已上传文件，避免垃圾任务目录堆积
        try:
            shutil.rmtree(jdir, ignore_errors=True)
        except Exception:
            pass
        return JSONResponse(
            {
                "ok": False,
                "error": "cannot_start",
                "fail_reason": reason,
                "message": reason,
                "admit": decision.value,
                "admit_reason": reason,
                "enhance": opt["enhance"],
                "worker": queue.worker_snapshot(),
            },
            status_code=503,
        )
    if not (opt["face"] or opt["plate"] or opt["text"] or opt["enhance"]):
        try:
            shutil.rmtree(jdir, ignore_errors=True)
        except Exception:
            pass
        return JSONResponse(
            {"ok": False, "error": "empty_options", "message": "至少启用打码目标之一或增强", "fail_reason": "empty_options"},
            status_code=400,
        )
    _save_options(jdir, opt, final_inp.name)
    _submit_redact_job(
        jid, jdir, final_inp, opt, admit=decision, admit_reason=reason
    )
    stage = "处理中" if decision == AdmitDecision.run else "排队中"
    return {
        "ok": True,
        "job_id": jid,
        "status": "queued" if decision == AdmitDecision.queue else "queued",
        "stage": stage,
        "message": reason,
        "admit": decision.value,
        "admit_reason": reason,
        "input": final_inp.name,
        "enhance": opt["enhance"],
        "enhance_only": bool(opt["enhance"] and not (opt["face"] or opt["plate"] or opt["text"])),
        "redact": {"face": opt["face"], "plate": opt["plate"], "text": opt["text"]},
        "worker": queue.worker_snapshot(),
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    st = queue.get(job_id)
    if not st:
        return JSONResponse({"error": "not found", "job_id": job_id}, status_code=404)
    return _status_payload(st)


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    ok, reason = queue.cancel(job_id)
    queue.reconcile()
    st = queue.get(job_id)
    if not st:
        return JSONResponse({"error": "not found", "job_id": job_id}, status_code=404)
    payload = _status_payload(st)
    payload["ok"] = ok
    payload["reason"] = reason
    return payload if ok else JSONResponse(payload, status_code=409)


@app.post("/api/jobs/{job_id}/pause")
def job_pause(job_id: str):
    ok, reason = queue.pause(job_id)
    queue.reconcile()
    st = queue.get(job_id)
    if not st:
        return JSONResponse({"error": "not found", "job_id": job_id}, status_code=404)
    payload = _status_payload(st)
    payload["ok"] = ok
    payload["reason"] = reason
    payload["message"] = st.message or "暂停将在下一帧安全点生效"
    return payload if ok else JSONResponse(payload, status_code=409)


@app.post("/api/jobs/{job_id}/resume")
def job_resume(job_id: str):
    ok, reason = queue.resume(job_id)
    queue.reconcile()
    st = queue.get(job_id)
    if not st:
        return JSONResponse({"error": "not found", "job_id": job_id}, status_code=404)
    payload = _status_payload(st)
    payload["ok"] = ok
    payload["reason"] = reason
    return payload if ok else JSONResponse(payload, status_code=409)


@app.post("/api/jobs/{job_id}/restart")
def job_restart(job_id: str):
    """用同一输入与 options.json 重新入队（新 job_id）。"""
    src = JOBS_DIR / job_id
    if not src.is_dir():
        return JSONResponse({"error": "not found", "job_id": job_id}, status_code=404)
    opt = _load_options(src)
    if not opt:
        return JSONResponse({"error": "options missing", "job_id": job_id}, status_code=400)
    inp = _find_input(src, opt.get("input_name"))
    if not inp or not inp.exists():
        return JSONResponse({"error": "input missing", "job_id": job_id}, status_code=400)

    old = queue.get(job_id)
    if old and old.status in (JobStatus.queued, JobStatus.running, JobStatus.paused):
        return JSONResponse(
            {"error": "job still active; cancel first", "job_id": job_id, "status": old.status.value},
            status_code=409,
        )

    opt_clean = {k: v for k, v in opt.items() if k != "input_name"}
    queue.reconcile()
    decision, reason = queue.admit(wants_enhance=bool(opt_clean.get("enhance")))
    if decision == AdmitDecision.reject:
        return JSONResponse(
            {
                "ok": False,
                "error": "cannot_start",
                "fail_reason": reason,
                "message": reason,
                "admit": decision.value,
                "admit_reason": reason,
                "source_job_id": job_id,
                "worker": queue.worker_snapshot(),
            },
            status_code=503,
        )

    new_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / new_id
    jdir.mkdir(parents=True, exist_ok=True)
    new_inp = jdir / inp.name
    shutil.copy2(inp, new_inp)
    _save_options(jdir, opt_clean, new_inp.name)
    _submit_redact_job(
        new_id, jdir, new_inp, opt_clean, admit=decision, admit_reason=reason
    )
    return {
        "ok": True,
        "job_id": new_id,
        "status": "queued",
        "stage": "处理中" if decision == AdmitDecision.run else "排队中",
        "message": f"已从 {job_id} 重启 · {reason}",
        "admit": decision.value,
        "admit_reason": reason,
        "source_job_id": job_id,
        "input": new_inp.name,
        "worker": queue.worker_snapshot(),
    }


@app.get("/api/worker")
def worker_status():
    """worker/进程存活与 GPU 空闲快照，供 UI 与调度判断。"""
    return queue.worker_snapshot()


@app.post("/api/scheduler/reconcile")
def scheduler_reconcile():
    """手动触发调度对账：清理 hung/dead，暴露卡住的 queued。"""
    return queue.reconcile()


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    st = queue.get(job_id)
    if not st or st.status != JobStatus.done or not st.output or not Path(st.output).exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(st.output, filename=f"{job_id}.mp4", media_type="video/mp4")
