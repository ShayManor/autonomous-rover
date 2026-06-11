"""Tiny thread-based background-job runner shared by calibration flows."""
import threading


class Job:
    def __init__(self, job_id, kind):
        self.id = job_id
        self.kind = kind
        self.status = "running"   # running | done | error
        self.progress = 0.0
        self.log = []
        self.result = None
        self.error = None

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "progress": self.progress, "log": list(self.log),
                "result": self.result, "error": self.error}


class JobRunner:
    """Run callables on daemon threads; poll status via get()/snapshot()."""

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._seq = 0

    def start(self, kind, target):
        """target(job) runs on a thread; set job.result / job.progress inside it."""
        with self._lock:
            self._seq += 1
            job = Job(self._seq, kind)
            self._jobs[job.id] = job

        def run():
            try:
                target(job)
                job.status = "done"
                job.progress = 1.0
            except Exception as e:  # surfaced to the UI, never crashes the server
                job.status = "error"
                job.error = str(e)

        threading.Thread(target=run, daemon=True).start()
        return job.id

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self):
        with self._lock:
            return {jid: j.as_dict() for jid, j in self._jobs.items()}
