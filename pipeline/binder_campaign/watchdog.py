"""Run a command, echo its output live, and stop it if it goes silent or runs too long. Pure Python (standard library).

Used for GPU jobs whose own log is the only sign of life: a hung job otherwise burns paid GPU time until a long timeout. A stall
is reported with the last lines the command printed, so the cause can be found instead of guessed.
"""
import subprocess
import threading
import time


class Stalled(RuntimeError):
    pass


def run_watched(command, stall_seconds, total_seconds, echo=print, poll=1.0):
    """Returns (returncode, output). Raises Stalled if nothing is printed for `stall_seconds` or the run exceeds `total_seconds`."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, state = [], {"last": time.monotonic()}

    def read():
        for line in process.stdout:
            state["last"] = time.monotonic()
            lines.append(line)
            echo(line.rstrip("\n"))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    started = time.monotonic()
    while process.poll() is None:
        time.sleep(poll)
        now = time.monotonic()
        reason = None
        if now - state["last"] > stall_seconds:
            reason = f"no output for {stall_seconds:.0f} s"
        elif now - started > total_seconds:
            reason = f"over the {total_seconds:.0f} s limit"
        if reason:
            process.kill()
            process.wait()
            reader.join(5)
            raise Stalled(f"Stopped: {reason}. Last output: " + "".join(lines)[-500:].replace("\n", " | "))
    reader.join(10)
    return process.returncode, "".join(lines)
