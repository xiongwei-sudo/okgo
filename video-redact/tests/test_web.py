"""上传先落盘再入队：相同文件名也不能共用路径。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import web.app as appmod  # noqa: E402
from pipeline.pipeline import JobResult  # noqa: E402
from pipeline.queue import JobState, JobStatus  # noqa: E402


class WebUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_jobs = appmod.JOBS
        appmod.JOBS = Path(self._tmp.name)
        self._pending: list[tuple] = []
        self._seen: list[tuple[Path, bytes, object]] = []

        def delayed_submit(fn, job_id=None):
            state = JobState(job_id=job_id, status=JobStatus.queued)
            appmod.queue._jobs[job_id] = state
            self._pending.append((fn, state))
            return job_id

        def fake_run(inp, out, weights, opts, progress=None):
            data = inp.read_bytes()
            self._seen.append((inp, data, opts))
            out.write_bytes(data)
            if progress:
                progress(1.0, "frame 1/1")
            return JobResult(output=out, frames=1, detections_total=0, notes=["N1【待补】"])

        self._old_submit = appmod.queue.submit
        self._old_run = appmod.run_redact
        appmod.queue.submit = delayed_submit
        appmod.run_redact = fake_run
        self.client = TestClient(appmod.app)

    def tearDown(self) -> None:
        appmod.queue.submit = self._old_submit
        appmod.run_redact = self._old_run
        appmod.JOBS = self._old_jobs
        self._tmp.cleanup()

    def _run_pending(self) -> None:
        for fn, state in self._pending:
            state.status = JobStatus.running
            fn(state)
            if state.status != JobStatus.failed:
                state.status = JobStatus.done
                state.progress = 1.0
        self._pending.clear()

    def test_same_filename_keeps_separate_bytes(self) -> None:
        payloads = [b"A" * 4000, b"B" * 5000]
        ids = []
        for payload in payloads:
            response = self.client.post(
                "/api/jobs",
                files={"file": ("same.mp4", payload, "video/mp4")},
                data={"face": "true", "mode": "gaussian", "scale": "2"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            ids.append(response.json()["job_id"])
        self.assertEqual(len(self._pending), 2)
        self._run_pending()
        self.assertEqual([item[1] for item in self._seen], payloads)
        paths = [item[0] for item in self._seen]
        self.assertNotEqual(paths[0], paths[1])
        for path, payload in zip(paths, payloads):
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "input.mp4")
            self.assertNotIn("pending", path.parts)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.parent.name, path.parent.name)
            self.assertTrue(path.parent.name.isalnum())
        for job_id, payload in zip(ids, payloads):
            status = self.client.get(f"/api/jobs/{job_id}")
            body = status.json()
            self.assertEqual(body["status"], "done")
            self.assertEqual(body["progress"], 1.0)
            self.assertIn("message", body)
            downloaded = self.client.get(f"/api/jobs/{job_id}/download")
            self.assertEqual(downloaded.content, payload)

    def test_path_traversal_filename_stays_in_job_dir(self) -> None:
        payload = b"xyz" * 100
        response = self.client.post(
            "/api/jobs",
            files={"file": ("../../etc/passwd.mp4", payload, "video/mp4")},
            data={"face": "true", "scale": "2"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self._run_pending()
        path, data, _opts = self._seen[-1]
        self.assertEqual(data, payload)
        self.assertEqual(path.name, "input.mp4")
        resolved = path.resolve()
        self.assertTrue(str(resolved).startswith(str(Path(self._tmp.name).resolve())))

    def test_enhance_flags_and_page(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        for name in ("stabilize", "lowlight", "denoise", "deblur", "superres", 'id="bar"'):
            self.assertIn(name, page.text)
        self.assertNotIn("CORSMiddleware", Path(appmod.__file__).read_text(encoding="utf-8"))
        response = self.client.post(
            "/api/jobs",
            files={"file": ("clip.mp4", b"1234", "video/mp4")},
            data={
                "face": "false",
                "plate": "true",
                "text": "false",
                "mode": "mosaic",
                "stabilize": "true",
                "lowlight": "true",
                "denoise": "true",
                "deblur": "true",
                "superres": "true",
                "scale": "4",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self._run_pending()
        opts = self._seen[-1][2]
        self.assertFalse(opts.face)
        self.assertTrue(opts.plate)
        self.assertEqual(opts.mode, "mosaic")
        self.assertTrue(opts.enhance.enable)
        self.assertTrue(opts.enhance.stabilize)
        self.assertTrue(opts.enhance.lowlight)
        self.assertTrue(opts.enhance.denoise)
        self.assertTrue(opts.enhance.deblur)
        self.assertTrue(opts.enhance.superres)
        self.assertEqual(opts.enhance.scale, 4)

    def test_empty_upload_rejected(self) -> None:
        response = self.client.post(
            "/api/jobs",
            files={"file": ("empty.mp4", b"", "video/mp4")},
            data={"face": "true", "scale": "2"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self._pending, [])


if __name__ == "__main__":
    unittest.main()
