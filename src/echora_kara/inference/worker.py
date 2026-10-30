"""Persistent JSON-lines worker for Echora Kara.

The process keeps acoustic models resident between jobs. Each request supplies
ordinary command-line arguments (``inference.cli``) and receives one JSON response on stdout.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import traceback

from echora_kara.inference.cli import main


def run() -> None:
    """Serve requests from stdin until EOF: each line is a JSON object with an ``argv`` string list, run through
    ``cli.main`` and answered by one JSON line on stdout: ``ok`` plus the captured ``logs`` (last 4000 characters),
    or ``ok: false`` with ``error`` and ``traceback`` (last 8000 characters)."""
    protocol_out = sys.stdout
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            argv = request.get("argv")
            if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                raise ValueError("worker request must contain a string argv list")
            logs = io.StringIO()
            with redirect_stdout(logs), redirect_stderr(logs):
                main(argv)
            response = {"ok": True, "logs": logs.getvalue()[-4000:]}
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc()[-8000:],
            }
        protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    run()
