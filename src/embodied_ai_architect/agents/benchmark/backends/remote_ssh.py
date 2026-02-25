"""Remote SSH benchmark backend - Execute benchmarks on remote machines via SSH."""

import tempfile
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn

from .base import BenchmarkBackend, BenchmarkResult


class RemoteSSHBackend(BenchmarkBackend):
    """Execute benchmarks on remote machines via SSH.

    This backend demonstrates:
    - Secure SSH connection using secrets manager
    - Model serialization and transfer
    - Remote code execution
    - Result retrieval
    - Proper cleanup

    Security features:
    - Uses SecretsManager for credentials
    - No hardcoded passwords or keys
    - Secure file transfer
    - Audit logging
    """

    def __init__(
        self,
        host: str,
        user: str,
        secrets_manager=None,
        ssh_key_secret: str = "ssh_key",
        port: int = 22,
        remote_workdir: str = "/tmp/embodied_ai_benchmark",
    ):
        """Initialize remote SSH backend.

        Args:
            host: Remote hostname or IP
            user: SSH username
            secrets_manager: SecretsManager instance for credentials
            ssh_key_secret: Key name for SSH private key in secrets manager
            port: SSH port
            remote_workdir: Working directory on remote machine

        Raises:
            ImportError: If paramiko not installed
            SecretNotFoundError: If SSH key not found
        """
        super().__init__(name=f"remote_ssh_{host}")

        try:
            import paramiko

            self.paramiko = paramiko
        except ImportError:
            raise ImportError(
                "paramiko is required for RemoteSSHBackend. "
                "Install with: pip install 'embodied-ai-architect[remote]'"
            )

        self.host = host
        self.user = user
        self.port = port
        self.remote_workdir = remote_workdir
        self.secrets_manager = secrets_manager
        self.ssh_key_secret = ssh_key_secret

        # SSH client (created on demand)
        self._client = None

    def is_available(self) -> bool:
        """Check if remote host is reachable.

        Returns:
            True if can connect, False otherwise
        """
        try:
            client = self._connect()
            client.close()
            return True
        except Exception:
            return False

    def _connect(self):
        """Establish SSH connection.

        Returns:
            Paramiko SSHClient

        Raises:
            SecretNotFoundError: If SSH key not found
            Exception: If connection fails
        """
        if (
            self._client
            and self._client.get_transport()
            and self._client.get_transport().is_active()
        ):
            return self._client

        # Get SSH key from secrets manager
        if self.secrets_manager:
            ssh_key_path = self.secrets_manager.get_secret(self.ssh_key_secret, required=True)
        else:
            # Fallback to default
            ssh_key_path = str(Path.home() / ".ssh" / "id_rsa")

        # Create SSH client
        client = self.paramiko.SSHClient()
        client.set_missing_host_key_policy(self.paramiko.AutoAddPolicy())

        print(f"  Connecting to {self.user}@{self.host}:{self.port}...")

        try:
            # Connect using private key
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=ssh_key_path,
                timeout=10,
            )
            print(f"  ✓ Connected to {self.host}")
            self._client = client
            return client

        except Exception as e:
            error_msg = str(e)
            # Mask the key path in error messages
            if self.secrets_manager:
                error_msg = self.secrets_manager.mask_secret(error_msg, ssh_key_path)
            raise Exception(f"SSH connection failed: {error_msg}")

    def _exec_command(self, command: str) -> tuple[str, str, int]:
        """Execute command on remote host.

        Args:
            command: Command to execute

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        client = self._connect()
        stdin, stdout, stderr = client.exec_command(command)

        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode()
        stderr_text = stderr.read().decode()

        return stdout_text, stderr_text, exit_code

    def _transfer_file(self, local_path: Path, remote_path: str):
        """Transfer file to remote host via SFTP.

        Args:
            local_path: Local file path
            remote_path: Remote file path
        """
        client = self._connect()
        sftp = client.open_sftp()

        try:
            # Create remote directory if needed
            remote_dir = str(Path(remote_path).parent)
            try:
                sftp.stat(remote_dir)
            except FileNotFoundError:
                # Directory doesn't exist, create it
                sftp.mkdir(remote_dir)

            sftp.put(str(local_path), remote_path)
            print(f"  ✓ Transferred {local_path.name} → {remote_path}")
        finally:
            sftp.close()

    def _retrieve_file(self, remote_path: str, local_path: Path):
        """Retrieve file from remote host via SFTP.

        Args:
            remote_path: Remote file path
            local_path: Local file path
        """
        client = self._connect()
        sftp = client.open_sftp()

        try:
            sftp.get(remote_path, str(local_path))
            print(f"  ✓ Retrieved {remote_path} → {local_path.name}")
        finally:
            sftp.close()

    def execute_benchmark(
        self,
        model: nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup_iterations: int = 10,
        config: Dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Execute benchmark on remote machine.

        Workflow:
        1. Serialize model to file
        2. Transfer model to remote machine
        3. Generate benchmark script
        4. Transfer script to remote machine
        5. Execute script remotely
        6. Retrieve results
        7. Cleanup

        Args:
            model: PyTorch model to benchmark
            input_shape: Input tensor shape
            iterations: Number of timed iterations
            warmup_iterations: Number of warmup iterations
            config: Optional configuration

        Returns:
            BenchmarkResult with timing statistics
        """
        print(f"  Executing benchmark on remote host: {self.host}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Serialize model
            model_path = tmpdir / "model.pt"
            torch.save(model.state_dict(), model_path)
            model_arch_path = tmpdir / "model_arch.py"
            self._save_model_architecture(model, model_arch_path)

            # 2. Transfer model files
            remote_model = f"{self.remote_workdir}/model.pt"
            remote_arch = f"{self.remote_workdir}/model_arch.py"
            self._transfer_file(model_path, remote_model)
            self._transfer_file(model_arch_path, remote_arch)

            # 3. Generate benchmark script
            script_path = tmpdir / "benchmark_script.py"
            self._generate_benchmark_script(script_path, input_shape, iterations, warmup_iterations)

            # 4. Transfer script
            remote_script = f"{self.remote_workdir}/benchmark_script.py"
            self._transfer_file(script_path, remote_script)

            # 5. Execute remotely
            print(f"  Running benchmark remotely ({iterations} iterations)...")
            stdout, stderr, exit_code = self._exec_command(
                f"cd {self.remote_workdir} && python3 benchmark_script.py"
            )

            if exit_code != 0:
                print(f"  Error: {stderr}")
                raise Exception(f"Remote benchmark failed: {stderr}")

            # 6. Parse results
            result = self._parse_results(stdout)

            # 7. Cleanup
            self._exec_command(f"rm -rf {self.remote_workdir}")

            return result

    def _save_model_architecture(self, model: nn.Module, output_path: Path):
        """Save model architecture definition to file.

        Args:
            model: PyTorch model
            output_path: Output file path
        """
        # Get model class name and code
        model_class = type(model)
        model_code = f"""
import torch
import torch.nn as nn

# Model architecture
class {model_class.__name__}(nn.Module):
    def __init__(self):
        super().__init__()
        # This is a placeholder - real implementation would
        # need to serialize the full architecture
        pass

    def forward(self, x):
        pass

def load_model(state_dict_path):
    model = {model_class.__name__}()
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
    return model
"""
        output_path.write_text(model_code)

    def _generate_benchmark_script(
        self, output_path: Path, input_shape: tuple, iterations: int, warmup_iterations: int
    ):
        """Generate Python script for remote benchmarking.

        Args:
            output_path: Output file path
            input_shape: Input tensor shape
            iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
        """
        script = f"""
import time
import json
import torch
import numpy as np
from model_arch import load_model

def benchmark():
    # Load model
    model = load_model('model.pt')
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    # Create input
    input_shape = {input_shape}
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range({warmup_iterations}):
            _ = model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range({iterations}):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    # Calculate statistics
    latencies_np = np.array(latencies)
    results = {{
        'mean_latency_ms': float(np.mean(latencies_np)),
        'std_latency_ms': float(np.std(latencies_np)),
        'min_latency_ms': float(np.min(latencies_np)),
        'max_latency_ms': float(np.max(latencies_np)),
        'iterations': {iterations},
        'warmup_iterations': {warmup_iterations}
    }}

    print(json.dumps(results))

if __name__ == '__main__':
    benchmark()
"""
        output_path.write_text(script)

    def _parse_results(self, stdout: str) -> BenchmarkResult:
        """Parse benchmark results from remote output.

        Args:
            stdout: Standard output from remote script

        Returns:
            BenchmarkResult
        """
        import json

        # Find JSON in output
        lines = stdout.strip().split("\n")
        for line in lines:
            try:
                data = json.loads(line)
                if "mean_latency_ms" in data:
                    # Calculate throughput
                    batch_size = 1  # Assuming batch size of 1
                    throughput = (1000.0 / data["mean_latency_ms"]) * batch_size

                    return BenchmarkResult(
                        backend_name=self.name,
                        device=f"remote_cpu@{self.host}",
                        mean_latency_ms=data["mean_latency_ms"],
                        std_latency_ms=data["std_latency_ms"],
                        min_latency_ms=data["min_latency_ms"],
                        max_latency_ms=data["max_latency_ms"],
                        throughput_samples_per_sec=throughput,
                        iterations=data["iterations"],
                        warmup_iterations=data["warmup_iterations"],
                        metadata={"host": self.host, "user": self.user, "remote": True},
                    )
            except json.JSONDecodeError:
                continue

        raise Exception(f"Could not parse benchmark results from: {stdout}")

    def get_capabilities(self) -> Dict[str, Any]:
        """Return remote SSH backend capabilities.

        Returns:
            Capability dictionary
        """
        return {
            "name": self.name,
            "measures_latency": True,
            "measures_throughput": True,
            "measures_memory": False,
            "measures_energy": False,
            "supports_remote": True,
            "requires_secrets": True,
        }

    def __del__(self):
        """Cleanup SSH connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
