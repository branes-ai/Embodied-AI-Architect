"""Power monitoring infrastructure for deployment validation.

Supports multiple power measurement methods:
- NVIDIA GPU: nvidia-smi, tegrastats (Jetson)
- Intel CPU: RAPL (Running Average Power Limit)
- Generic: psutil-based estimation

Power measurement is critical for embodied AI where energy is a functional
requirement - systems that exceed their power budget simply don't work.
"""

import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class PowerSample:
    """Single power measurement sample."""

    timestamp: float  # Unix timestamp
    total_watts: float  # Total system/device power
    gpu_watts: float | None = None  # GPU-specific power
    cpu_watts: float | None = None  # CPU-specific power
    memory_watts: float | None = None  # Memory power if available


@dataclass
class PowerMeasurement:
    """Aggregated power measurement over a duration."""

    samples: list[PowerSample] = field(default_factory=list)
    duration_sec: float = 0.0
    method: str = "unknown"

    @property
    def mean_watts(self) -> float:
        """Mean power consumption in watts."""
        if not self.samples:
            return 0.0
        return float(np.mean([s.total_watts for s in self.samples]))

    @property
    def std_watts(self) -> float:
        """Standard deviation of power in watts."""
        if len(self.samples) < 2:
            return 0.0
        return float(np.std([s.total_watts for s in self.samples]))

    @property
    def peak_watts(self) -> float:
        """Peak power consumption in watts."""
        if not self.samples:
            return 0.0
        return float(np.max([s.total_watts for s in self.samples]))

    @property
    def mean_gpu_watts(self) -> float | None:
        """Mean GPU power if available."""
        gpu_samples = [s.gpu_watts for s in self.samples if s.gpu_watts is not None]
        if not gpu_samples:
            return None
        return float(np.mean(gpu_samples))

    @property
    def mean_cpu_watts(self) -> float | None:
        """Mean CPU power if available."""
        cpu_samples = [s.cpu_watts for s in self.samples if s.cpu_watts is not None]
        if not cpu_samples:
            return None
        return float(np.mean(cpu_samples))


class PowerMonitor(ABC):
    """Abstract base class for power monitoring."""

    def __init__(self, name: str):
        self.name = name
        self._sampling = False
        self._samples: list[PowerSample] = []
        self._sample_thread: threading.Thread | None = None
        self._sample_interval: float = 0.1  # 100ms default

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this power monitor is available on the system."""
        pass

    @abstractmethod
    def _read_power(self) -> PowerSample:
        """Read current power consumption. Override in subclass."""
        pass

    def start_sampling(self, interval_sec: float = 0.1) -> None:
        """Start background power sampling."""
        if self._sampling:
            return

        self._sample_interval = interval_sec
        self._samples = []
        self._sampling = True
        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()

    def stop_sampling(self) -> PowerMeasurement:
        """Stop sampling and return aggregated measurements."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None

        duration = 0.0
        if len(self._samples) >= 2:
            duration = self._samples[-1].timestamp - self._samples[0].timestamp

        return PowerMeasurement(
            samples=self._samples.copy(),
            duration_sec=duration,
            method=self.name,
        )

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        while self._sampling:
            try:
                sample = self._read_power()
                self._samples.append(sample)
            except Exception:
                pass  # Skip failed samples
            time.sleep(self._sample_interval)

    def measure_during(
        self,
        workload: Callable[[], None],
        warmup_iterations: int = 10,
        measurement_iterations: int = 50,
    ) -> PowerMeasurement:
        """Measure power while running a workload.

        Args:
            workload: Function to call repeatedly during measurement
            warmup_iterations: Number of warmup calls before measuring
            measurement_iterations: Number of calls during measurement

        Returns:
            PowerMeasurement with samples collected during workload execution
        """
        # Warmup
        for _ in range(warmup_iterations):
            workload()

        # Start sampling
        self.start_sampling(interval_sec=0.05)  # 50ms sampling

        start_time = time.perf_counter()

        # Run workload during measurement
        for _ in range(measurement_iterations):
            workload()

        elapsed = time.perf_counter() - start_time

        # Stop and get results
        measurement = self.stop_sampling()
        measurement.duration_sec = elapsed

        return measurement


class NvidiaSmiMonitor(PowerMonitor):
    """Power monitoring via nvidia-smi for discrete NVIDIA GPUs."""

    def __init__(self):
        super().__init__("nvidia-smi")
        self._nvidia_smi_path: str | None = None

    def is_available(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._nvidia_smi_path = "nvidia-smi"
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return False

    def _read_power(self) -> PowerSample:
        """Read GPU power from nvidia-smi."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        gpu_power = float(result.stdout.strip().split("\n")[0])

        return PowerSample(
            timestamp=time.time(),
            total_watts=gpu_power,
            gpu_watts=gpu_power,
        )


class TegrastatsMonitor(PowerMonitor):
    """Power monitoring via tegrastats for NVIDIA Jetson devices.

    Tegrastats provides detailed power information for Jetson platforms:
    - VDD_IN: Total board input power
    - VDD_CPU_GPU_CV: CPU/GPU/CV power rail
    - VDD_SOC: SoC power
    """

    def __init__(self):
        super().__init__("tegrastats")
        self._power_paths: dict[str, Path] = {}

    def is_available(self) -> bool:
        """Check if running on Jetson with power monitoring."""
        # Check for Jetson power sysfs entries
        power_base = Path("/sys/bus/i2c/drivers/ina3221x")

        if not power_base.exists():
            # Try alternative path for newer Jetson
            power_base = Path("/sys/class/hwmon")

        # Look for power measurement files
        for hwmon in power_base.glob("hwmon*"):
            for sensor in hwmon.glob("in*_input"):
                self._power_paths[sensor.stem] = sensor

        # Also check for tegrastats binary
        try:
            result = subprocess.run(
                ["which", "tegrastats"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return len(self._power_paths) > 0

    def _read_power(self) -> PowerSample:
        """Read power from Jetson power rails."""
        total_power = 0.0
        gpu_power = None
        cpu_power = None

        # Try reading from sysfs power files
        for name, path in self._power_paths.items():
            try:
                value = float(path.read_text().strip()) / 1000.0  # Convert mW to W
                total_power += value
                if "gpu" in name.lower():
                    gpu_power = value
                elif "cpu" in name.lower():
                    cpu_power = value
            except (ValueError, IOError):
                pass

        # Fallback: parse tegrastats output
        if total_power == 0.0:
            try:
                result = subprocess.run(
                    ["tegrastats", "--interval", "100", "--stop", "1"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                # Parse VDD_IN from output
                for part in result.stdout.split():
                    if "VDD_IN" in part or "POM_5V_IN" in part:
                        # Format: VDD_IN 5000mW/5000mW or similar
                        power_str = part.split()[1].split("/")[0]
                        total_power = float(power_str.replace("mW", "")) / 1000.0
                        break
            except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                pass

        return PowerSample(
            timestamp=time.time(),
            total_watts=total_power,
            gpu_watts=gpu_power,
            cpu_watts=cpu_power,
        )


class IntelRaplMonitor(PowerMonitor):
    """Power monitoring via Intel RAPL (Running Average Power Limit).

    RAPL provides accurate power measurements for Intel CPUs via MSRs
    exposed through the powercap sysfs interface.
    """

    def __init__(self):
        super().__init__("intel-rapl")
        self._rapl_base = Path("/sys/class/powercap/intel-rapl")
        self._energy_files: dict[str, Path] = {}
        self._last_energy: dict[str, float] = {}
        self._last_time: float = 0.0

    def is_available(self) -> bool:
        """Check if Intel RAPL is available."""
        if not self._rapl_base.exists():
            return False

        # Find energy measurement files
        for domain in self._rapl_base.glob("intel-rapl:*"):
            energy_file = domain / "energy_uj"
            if energy_file.exists():
                name = domain.name
                self._energy_files[name] = energy_file

        return len(self._energy_files) > 0

    def _read_power(self) -> PowerSample:
        """Read power from RAPL energy counters."""
        current_time = time.time()
        current_energy: dict[str, float] = {}

        total_power = 0.0
        cpu_power = 0.0

        for name, path in self._energy_files.items():
            try:
                energy_uj = float(path.read_text().strip())
                current_energy[name] = energy_uj

                if name in self._last_energy and self._last_time > 0:
                    delta_energy = energy_uj - self._last_energy[name]
                    delta_time = current_time - self._last_time

                    if delta_energy < 0:  # Counter wrapped
                        delta_energy += 2**32  # Assume 32-bit counter

                    power_w = (delta_energy / 1_000_000.0) / delta_time  # uJ to W
                    total_power += power_w

                    if "package" in name.lower() or "core" in name.lower():
                        cpu_power += power_w

            except (ValueError, IOError):
                pass

        self._last_energy = current_energy
        self._last_time = current_time

        return PowerSample(
            timestamp=current_time,
            total_watts=total_power,
            cpu_watts=cpu_power if cpu_power > 0 else None,
        )


class PsutilMonitor(PowerMonitor):
    """Fallback power estimation using psutil CPU usage.

    This is a rough estimation based on CPU utilization and TDP.
    Should only be used when hardware power monitoring is unavailable.
    """

    def __init__(self, cpu_tdp_watts: float = 65.0):
        super().__init__("psutil-estimate")
        self._cpu_tdp = cpu_tdp_watts
        self._psutil = None

    def is_available(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil

            self._psutil = psutil
            return True
        except ImportError:
            return False

    def _read_power(self) -> PowerSample:
        """Estimate power from CPU utilization."""
        if self._psutil is None:
            return PowerSample(timestamp=time.time(), total_watts=0.0)

        # Get CPU utilization (0-100)
        cpu_percent = self._psutil.cpu_percent(interval=0.1)

        # Rough power estimation: idle ~10% TDP, scales with usage
        idle_power = self._cpu_tdp * 0.1
        active_power = self._cpu_tdp * 0.9 * (cpu_percent / 100.0)
        estimated_power = idle_power + active_power

        return PowerSample(
            timestamp=time.time(),
            total_watts=estimated_power,
            cpu_watts=estimated_power,
        )


def get_power_monitor() -> PowerMonitor | None:
    """Get the best available power monitor for the current system.

    Returns:
        PowerMonitor instance or None if no monitoring available
    """
    # Try monitors in order of preference
    monitors = [
        TegrastatsMonitor(),  # Jetson (most accurate for edge)
        NvidiaSmiMonitor(),  # Discrete NVIDIA GPU
        IntelRaplMonitor(),  # Intel CPU
        PsutilMonitor(),  # Fallback estimation
    ]

    for monitor in monitors:
        if monitor.is_available():
            return monitor

    return None
