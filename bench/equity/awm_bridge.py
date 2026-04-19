"""Python bridge to the TypeScript AWM CLI.

Launches a single long-running `tsx awm_cli.ts` process and communicates
via a NDJSON pipe on stdin/stdout. One subprocess is reused across all
predict/record/reset calls so we don't pay Node start-up per observation.

The AWM monorepo root (containing node_modules/.bin/tsx and the workspace
symlinks) is used as the subprocess cwd so that `@heybeaux/awm-core` and
`@heybeaux/awm-equity-store` imports resolve.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


MONOREPO_ROOT = Path(__file__).resolve().parents[2]  # ~/awm
CLI_SCRIPT = Path(__file__).resolve().parent / "awm_cli.ts"
TSX_BIN = MONOREPO_ROOT / "node_modules" / ".bin" / "tsx"


class AWMBridge:
    """Long-lived subprocess wrapper around the AWM TypeScript CLI."""

    def __init__(
        self,
        db_path: str = ":memory:",
        call_timeout_sec: float = 30.0,
        startup_timeout_sec: float = 15.0,
    ) -> None:
        if not TSX_BIN.exists():
            raise RuntimeError(
                f"tsx binary not found at {TSX_BIN}. "
                f"Run `npm install` at {MONOREPO_ROOT}."
            )
        if not CLI_SCRIPT.exists():
            raise RuntimeError(f"awm_cli.ts not found at {CLI_SCRIPT}")

        env = os.environ.copy()
        env["AWM_DB_PATH"] = db_path
        env["NODE_NO_WARNINGS"] = "1"

        self._proc = subprocess.Popen(
            [str(TSX_BIN), str(CLI_SCRIPT)],
            cwd=str(MONOREPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._lock = threading.Lock()
        self._timeout = call_timeout_sec
        self._closed = False

        # Drain stderr in the background so a warning doesn't block the pipe.
        self._stderr_buf: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Quick health check: if the subprocess dies before first call, raise.
        t0 = time.time()
        while time.time() - t0 < 0.5:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"awm_cli.ts exited immediately: "
                    f"{self._collect_stderr()!r}"
                )
            time.sleep(0.05)

    # ─── internals ────────────────────────────────────────────────

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        try:
            for line in self._proc.stderr:
                self._stderr_buf.append(line)
                # Forward to our stderr so the operator sees CLI errors.
                sys.stderr.write(f"[awm_cli] {line}")
                sys.stderr.flush()
        except Exception:
            pass

    def _collect_stderr(self) -> str:
        return "".join(self._stderr_buf[-20:])

    def _call(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("AWMBridge is closed")
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"awm_cli.ts exited (code={self._proc.returncode}): "
                f"{self._collect_stderr()!r}"
            )

        payload = json.dumps(msg) + "\n"
        with self._lock:
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except BrokenPipeError as e:
                raise RuntimeError(
                    f"awm_cli.ts pipe closed while writing: "
                    f"{self._collect_stderr()!r}"
                ) from e

            # readline has no timeout natively; use a watchdog thread.
            result: dict[str, Any] = {}
            exc: list[BaseException] = []

            def _read() -> None:
                try:
                    line = self._proc.stdout.readline()  # type: ignore[union-attr]
                    if not line:
                        raise RuntimeError(
                            "awm_cli.ts closed stdout without responding"
                        )
                    result.update(json.loads(line))
                except BaseException as e:  # noqa: BLE001
                    exc.append(e)

            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(self._timeout)
            if t.is_alive():
                raise TimeoutError(
                    f"awm_cli.ts did not respond within {self._timeout}s"
                )
            if exc:
                raise exc[0]
            return result

    # ─── public API ────────────────────────────────────────────────

    def predict(
        self,
        ticker: str,
        regime: str,
        features: dict[str, float] | None = None,  # unused; kept for spec parity
        embedding: list[float] | None = None,       # unused; kept for spec parity
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        tid = trace_id or uuid4().hex
        return self._call(
            {
                "action": "predict",
                "ticker": ticker,
                "regime": regime,
                "trace_id": tid,
            }
        )

    def record(
        self,
        ticker: str,
        regime: str,
        outcome: int | bool,
        trace_id: str | None = None,
    ) -> None:
        tid = trace_id or uuid4().hex
        resp = self._call(
            {
                "action": "record",
                "ticker": ticker,
                "regime": regime,
                "outcome": int(bool(outcome)),
                "trace_id": tid,
            }
        )
        if not resp.get("ok", False):
            raise RuntimeError(f"record failed: {resp}")

    def reset(self) -> None:
        resp = self._call({"action": "reset"})
        if not resp.get("ok", False):
            raise RuntimeError(f"reset failed: {resp}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                try:
                    self._call({"action": "shutdown"})
                except Exception:
                    pass
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.send_signal(signal.SIGTERM)
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            pass

    def __enter__(self) -> "AWMBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ─── smoke test ──────────────────────────────────────────────────

def _smoke() -> int:
    print("[awm_bridge] starting subprocess...", flush=True)
    with AWMBridge() as bridge:
        p0 = bridge.predict("AAPL", "trending_up")
        print(f"[awm_bridge] prior AAPL/trending_up: {p0}", flush=True)
        assert 0.4 < p0["p_up"] < 0.8, p0
        for _ in range(5):
            bridge.record("AAPL", "trending_up", 1)
        for _ in range(2):
            bridge.record("AAPL", "trending_up", 0)
        p1 = bridge.predict("AAPL", "trending_up")
        print(f"[awm_bridge] post-updates AAPL/trending_up: {p1}", flush=True)
        assert p1["observations"] == 7, p1
        assert p1["p_up"] > p0["p_up"], (p0, p1)
        bridge.reset()
        p2 = bridge.predict("AAPL", "trending_up")
        print(f"[awm_bridge] after reset: {p2}", flush=True)
        assert p2["observations"] == 0, p2
    print("[awm_bridge] smoke test OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
