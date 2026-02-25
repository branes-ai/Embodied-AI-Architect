"""Power monitoring for benchmarks.

Supports AMD (ryzen_smu), Intel (RAPL), and external measurement.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import statistics
import subprocess
import threading
import time


@dataclass
class PowerMetrics:
    """Power measurement results."""

    mean_watts: float
    peak_watts: float
    min_watts: float
    samples: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    energy_j: float = 0.0  # Total energy in joules

    @classmethod
    def from_samples(cls, samples: list[float], interval_ms: float) -> "PowerMetrics":
        """Create metrics from power samples.

        Args:
            samples: Power samples in watts
            interval_ms: Sampling interval in milliseconds
        """
        if not samples:
            return cls(0.0, 0.0, 0.0)

        duration_s = len(samples) * interval_ms / 1000.0
        energy_j = sum(samples) * interval_ms / 1000.0

        return cls(
            mean_watts=statistics.mean(samples),
            peak_watts=max(samples),
            min_watts=min(samples),
            samples=samples,
            duration_s=duration_s,
            energy_j=energy_j,
        )


class PowerMonitor:
    """Monitor power consumption during benchmarks.

    Supports multiple backends:
    - RAPL: Intel Running Average Power Limit (Linux)
    - AMD SMU: AMD System Management Unit (requires ryzen_smu)
    - External: Manual or external power meter
    """

    def __init__(self, backend: str = "auto", interval_ms: float = 100):
        """Initialize power monitor.

        Args:
            backend: Power monitoring backend (auto, rapl, amd_smu, external)
            interval_ms: Sampling interval in milliseconds
        """
        self.interval_ms = interval_ms
        self.backend = backend
        self._monitor = None
        self._thread = None
        self._samples: list[float] = []
        self._running = False

        if backend == "auto":
            self._monitor = self._detect_backend()
        elif backend == "rapl":
            self._monitor = RAPLMonitor()
        elif backend == "amd_smu":
            self._monitor = AMDSMUMonitor()
        elif backend == "external":
            self._monitor = ExternalMonitor()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _detect_backend(self):
        """Auto-detect available power monitoring backend."""
        # Try RAPL first (most common on Linux)
        if RAPLMonitor.is_available():
            print("[PowerMonitor] Using RAPL backend")
            return RAPLMonitor()

        # Try AMD SMU
        if AMDSMUMonitor.is_available():
            print("[PowerMonitor] Using AMD SMU backend")
            return AMDSMUMonitor()

        print("[PowerMonitor] No power monitoring available, using stub")
        return StubMonitor()

    def start(self) -> None:
        """Start power monitoring in background thread."""
        if self._running:
            return

        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        """Background sampling loop."""
        while self._running:
            try:
                power = self._monitor.read_power()
                if power is not None:
                    self._samples.append(power)
            except Exception as e:
                print(f"[PowerMonitor] Sampling error: {e}")

            time.sleep(self.interval_ms / 1000.0)

    def stop(self) -> PowerMetrics:
        """Stop monitoring and return metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        return PowerMetrics.from_samples(self._samples, self.interval_ms)

    def is_available(self) -> bool:
        """Check if power monitoring is available."""
        return not isinstance(self._monitor, StubMonitor)


class RAPLMonitor:
    """Intel RAPL power monitoring via sysfs."""

    RAPL_PATH = Path("/sys/class/powercap/intel-rapl")

    def __init__(self):
        self.package_path = None
        self._last_energy = None
        self._last_time = None

        # Find package-0 (main CPU package)
        if self.RAPL_PATH.exists():
            for domain in self.RAPL_PATH.iterdir():
                if domain.name.startswith("intel-rapl:"):
                    name_file = domain / "name"
                    if name_file.exists():
                        name = name_file.read_text().strip()
                        if name == "package-0":
                            self.package_path = domain / "energy_uj"
                            break

    @classmethod
    def is_available(cls) -> bool:
        """Check if RAPL is available."""
        if not cls.RAPL_PATH.exists():
            return False

        # Check if we can read energy
        for domain in cls.RAPL_PATH.iterdir():
            if domain.name.startswith("intel-rapl:"):
                energy_file = domain / "energy_uj"
                if energy_file.exists():
                    try:
                        energy_file.read_text()
                        return True
                    except PermissionError:
                        return False
        return False

    def read_power(self) -> float | None:
        """Read instantaneous power from RAPL."""
        if not self.package_path:
            return None

        try:
            energy_uj = int(self.package_path.read_text().strip())
            current_time = time.time()

            if self._last_energy is not None and self._last_time is not None:
                delta_energy = energy_uj - self._last_energy
                delta_time = current_time - self._last_time

                # Handle counter wraparound
                if delta_energy < 0:
                    max_energy = int(
                        (self.package_path.parent / "max_energy_range_uj").read_text().strip()
                    )
                    delta_energy += max_energy

                if delta_time > 0:
                    power_w = (delta_energy / 1e6) / delta_time
                    self._last_energy = energy_uj
                    self._last_time = current_time
                    return power_w

            self._last_energy = energy_uj
            self._last_time = current_time
            return None

        except Exception:
            return None


class AMDSMUMonitor:
    """AMD SMU power monitoring.

    Requires ryzen_smu kernel module and ryzen_monitor tool.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if AMD SMU monitoring is available."""
        # Check for ryzen_monitor tool
        try:
            result = subprocess.run(
                ["which", "ryzen_monitor"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def read_power(self) -> float | None:
        """Read power from AMD SMU."""
        try:
            # Use ryzen_monitor to get power
            result = subprocess.run(
                ["ryzen_monitor", "-p"],
                capture_output=True,
                text=True,
                timeout=1,
            )

            if result.returncode == 0:
                # Parse output for package power
                for line in result.stdout.split("\n"):
                    if "Package Power" in line or "PPT" in line:
                        # Extract number
                        parts = line.split()
                        for part in parts:
                            try:
                                return float(part.replace("W", ""))
                            except ValueError:
                                continue

            return None

        except Exception:
            return None


class ExternalMonitor:
    """Placeholder for external power measurement.

    Can be extended to support:
    - USB power meters (e.g., Power-Z)
    - Smart PDUs
    - Lab equipment via SCPI
    """

    @classmethod
    def is_available(cls) -> bool:
        return True  # Always available as placeholder

    def read_power(self) -> float | None:
        """External measurement not implemented."""
        return None


class StubMonitor:
    """Stub monitor when no backend is available."""

    @classmethod
    def is_available(cls) -> bool:
        return True

    def read_power(self) -> float | None:
        return None


def get_power_monitor(backend: str = "auto", interval_ms: float = 100) -> PowerMonitor:
    """Factory function to create power monitor.

    Args:
        backend: Backend to use (auto, rapl, amd_smu, external)
        interval_ms: Sampling interval

    Returns:
        Configured PowerMonitor instance
    """
    return PowerMonitor(backend=backend, interval_ms=interval_ms)
