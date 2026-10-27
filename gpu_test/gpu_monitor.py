"""Background GPU utilization/power sampler, via polling nvidia-smi.

Note: memory.used / memory.total are NOT queried here -- confirmed
empirically that this machine's driver reports them as [N/A] (a known GB10
driver limitation, not a bug in this script). GPU memory footprint per
model is instead read from each backend's own reporting (vLLM's startup
log line, `ollama ps`) in the sweep scripts.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Optional


class GpuMonitor:
    def __init__(self, interval_s: float = 0.3):
        self.interval_s = interval_s
        self._samples: list[tuple[float, float]] = []  # (util_pct, power_w)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample_once(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            util_str, power_str = [x.strip() for x in out.split(",")]
            self._samples.append((float(util_str), float(power_str)))
        except Exception:
            pass  # a missed sample shouldn't kill the benchmark

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self._samples:
            return {"gpu_util_pct_mean": None, "gpu_util_pct_max": None,
                     "power_w_mean": None, "power_w_max": None}
        utils = [s[0] for s in self._samples]
        powers = [s[1] for s in self._samples]
        return {
            "gpu_util_pct_mean": round(sum(utils) / len(utils), 1),
            "gpu_util_pct_max": max(utils),
            "power_w_mean": round(sum(powers) / len(powers), 2),
            "power_w_max": max(powers),
        }
