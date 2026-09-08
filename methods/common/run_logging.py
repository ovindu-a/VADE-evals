"""Live-progress logging for a run/sweep -- tail -f a file under logs/ to
monitor a script that's running detached (tmux/nohup/a background job)
instead of having to remember to redirect its stdout by hand every time.

tee_to_log wraps a script's whole body: every print() (from that script
AND anything it calls -- train_layer/eval_layer/score_file all use plain
print()) is duplicated to both the real stdout/stderr and the log file, so
running interactively still shows output normally, while a detached run
always has a canonical log file under logs/ to watch. No changes needed
anywhere else -- this is stdout-level, not a logger threaded through every
function signature.
"""
import os
import sys
from contextlib import contextmanager


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


@contextmanager
def tee_to_log(log_path):
    """Creates log_path's directory if missing, opens it in append mode
    (so a resumed run keeps its history), and duplicates stdout+stderr into
    it for the duration of the `with` block."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_f = open(log_path, "a")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(old_stdout, log_f)
    sys.stderr = Tee(old_stderr, log_f)
    try:
        print(f"\n=== logging to {log_path} ===", flush=True)
        yield log_path
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_f.close()
