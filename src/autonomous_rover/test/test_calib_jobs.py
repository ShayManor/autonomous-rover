import time
from autonomous_rover.nodes.master.calibration.jobs import JobRunner


def _wait(runner, jid, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if runner.get(jid).status != "running":
            return
        time.sleep(0.01)


def test_job_success_sets_result_and_progress():
    r = JobRunner()
    jid = r.start("demo", lambda job: job.__setattr__("result", {"ok": True}))
    _wait(r, jid)
    job = r.get(jid)
    assert job.status == "done"
    assert job.progress == 1.0
    assert job.result == {"ok": True}


def test_job_error_is_captured():
    r = JobRunner()
    def boom(job):
        raise RuntimeError("nope")
    jid = r.start("demo", boom)
    _wait(r, jid)
    job = r.get(jid)
    assert job.status == "error"
    assert "nope" in job.error


def test_snapshot_lists_jobs():
    r = JobRunner()
    jid = r.start("demo", lambda job: None)
    _wait(r, jid)
    snap = r.snapshot()
    assert jid in snap
    assert snap[jid]["kind"] == "demo"
