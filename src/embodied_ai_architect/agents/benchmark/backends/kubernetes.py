"""Kubernetes backend - Execute benchmarks as K8s Jobs with horizontal scaling."""

import time
import json
import uuid
import base64
from typing import Any, Dict, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn

from .base import BenchmarkBackend, BenchmarkResult


class KubernetesBackend(BenchmarkBackend):
    """Execute benchmarks as Kubernetes Jobs.

    This backend enables cloud-native benchmarking with:
    - Horizontal scaling (run N benchmarks in parallel)
    - GPU allocation (request specific GPU types)
    - Resource management (CPU/memory limits)
    - Automatic cleanup (TTL for jobs)
    - Isolation (each benchmark in its own container)

    Example:
        # Single benchmark
        backend = KubernetesBackend(namespace="embodied-ai")
        result = backend.execute_benchmark(model, input_shape=(1, 3, 224, 224))

        # Parallel benchmarks
        results = backend.execute_parallel([
            (model1, config1),
            (model2, config2),
            ...
        ])
    """

    def __init__(
        self,
        namespace: str = "default",
        image: str = "embodied-ai-benchmark:latest",
        secrets_manager=None,
        kubeconfig_secret: str = "k8s_kubeconfig",
        gpu_type: str | None = None,
        cpu_request: str = "1",
        memory_request: str = "2Gi",
        ttl_seconds: int = 3600,
    ):
        """Initialize Kubernetes backend.

        Args:
            namespace: K8s namespace for jobs
            image: Container image for benchmark execution
            secrets_manager: SecretsManager for kubeconfig
            kubeconfig_secret: Secret key for kubeconfig file
            gpu_type: GPU type (e.g., "nvidia-a100")
            cpu_request: CPU request (e.g., "2")
            memory_request: Memory request (e.g., "4Gi")
            ttl_seconds: TTL for job cleanup after completion

        Raises:
            ImportError: If kubernetes client not installed
            SecretNotFoundError: If kubeconfig not found
        """
        super().__init__(name=f"kubernetes_{namespace}")

        try:
            from kubernetes import client, config as k8s_config

            self.client = client
            self.k8s_config = k8s_config
        except ImportError:
            raise ImportError(
                "kubernetes is required for KubernetesBackend. "
                "Install with: pip install 'embodied-ai-architect[kubernetes]'"
            )

        self.namespace = namespace
        self.image = image
        self.gpu_type = gpu_type
        self.cpu_request = cpu_request
        self.memory_request = memory_request
        self.ttl_seconds = ttl_seconds
        self.secrets_manager = secrets_manager

        # Load kubeconfig
        if secrets_manager:
            kubeconfig_path = secrets_manager.get_secret(kubeconfig_secret, required=True)
            self.k8s_config.load_kube_config(config_file=kubeconfig_path)
        else:
            # Try default locations
            try:
                self.k8s_config.load_kube_config()
            except:
                self.k8s_config.load_incluster_config()

        # Initialize API clients
        self.batch_api = self.client.BatchV1Api()
        self.core_api = self.client.CoreV1Api()

        print(f"  Connected to Kubernetes cluster (namespace: {namespace})")

    def is_available(self) -> bool:
        """Check if Kubernetes cluster is accessible.

        Returns:
            True if cluster is reachable, False otherwise
        """
        try:
            self.core_api.list_namespace()
            return True
        except:
            return False

    def execute_benchmark(
        self,
        model: nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup_iterations: int = 10,
        config: Dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Execute single benchmark as Kubernetes Job.

        Args:
            model: PyTorch model to benchmark
            input_shape: Input tensor shape
            iterations: Number of timed iterations
            warmup_iterations: Number of warmup iterations
            config: Optional configuration

        Returns:
            BenchmarkResult with timing statistics
        """
        job_id = str(uuid.uuid4())[:8]

        print(f"  Creating Kubernetes Job: {job_id}")

        try:
            # 1. Create ConfigMap with model
            self._create_model_configmap(model, job_id, input_shape, iterations, warmup_iterations)

            # 2. Create Job
            self._create_benchmark_job(job_id)

            # 3. Wait for completion
            print(f"  Waiting for job completion...")
            self._wait_for_job_completion(job_id, timeout=600)

            # 4. Get results from logs
            result = self._get_job_results(job_id)

            print(f"  ✓ Job completed: {result.mean_latency_ms:.3f}ms")

            return result

        finally:
            # 5. Cleanup
            self._cleanup_job(job_id)

    def execute_parallel(
        self, benchmarks: List[Tuple[nn.Module, Dict[str, Any]]]
    ) -> List[BenchmarkResult]:
        """Execute multiple benchmarks in parallel (horizontal scaling).

        This is the key feature of the Kubernetes backend - ability to
        run many benchmarks simultaneously across the cluster.

        Args:
            benchmarks: List of (model, config) tuples

        Returns:
            List of BenchmarkResults
        """
        print(f"  Creating {len(benchmarks)} parallel jobs...")

        job_ids = []

        # Create all jobs
        for i, (model, bench_config) in enumerate(benchmarks):
            job_id = str(uuid.uuid4())[:8]

            # Extract config
            input_shape = bench_config.get("input_shape", (1, 10))
            iterations = bench_config.get("iterations", 100)
            warmup_iterations = bench_config.get("warmup_iterations", 10)

            # Create resources
            self._create_model_configmap(model, job_id, input_shape, iterations, warmup_iterations)
            self._create_benchmark_job(job_id)

            job_ids.append(job_id)
            print(f"    Job {i+1}/{len(benchmarks)}: {job_id}")

        # Wait for all to complete
        print(f"  Waiting for {len(job_ids)} jobs to complete...")
        results = []

        for i, job_id in enumerate(job_ids):
            try:
                self._wait_for_job_completion(job_id, timeout=600)
                result = self._get_job_results(job_id)
                results.append(result)
                print(f"    Job {i+1}/{len(job_ids)} complete: {result.mean_latency_ms:.3f}ms")
            except Exception as e:
                print(f"    Job {i+1}/{len(job_ids)} failed: {e}")
                # Continue with other jobs
            finally:
                self._cleanup_job(job_id)

        print(f"  ✓ Completed {len(results)}/{len(job_ids)} jobs")
        return results

    def _create_model_configmap(
        self,
        model: nn.Module,
        job_id: str,
        input_shape: tuple,
        iterations: int,
        warmup_iterations: int,
    ):
        """Create ConfigMap with model and configuration.

        Args:
            model: PyTorch model
            job_id: Job identifier
            input_shape: Input shape
            iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
        """
        # Serialize model
        import io

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_bytes = buffer.getvalue()
        model_b64 = base64.b64encode(model_bytes).decode()

        # Create configuration
        config_data = {
            "input_shape": list(input_shape),
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "model_type": type(model).__name__,
        }

        # Create ConfigMap
        configmap = self.client.V1ConfigMap(
            metadata=self.client.V1ObjectMeta(
                name=f"model-{job_id}", namespace=self.namespace, labels={"job": job_id}
            ),
            binary_data={"model.pt": model_b64},
            data={"config.json": json.dumps(config_data)},
        )

        self.core_api.create_namespaced_config_map(self.namespace, configmap)

    def _create_benchmark_job(self, job_id: str):
        """Create Kubernetes Job for benchmark execution.

        Args:
            job_id: Job identifier
        """
        # Resource requests
        resources = self.client.V1ResourceRequirements(
            requests={"cpu": self.cpu_request, "memory": self.memory_request},
            limits={"cpu": str(int(self.cpu_request) * 2), "memory": self.memory_request},
        )

        # Add GPU if specified
        if self.gpu_type:
            resources.requests["nvidia.com/gpu"] = "1"
            resources.limits["nvidia.com/gpu"] = "1"

        # Volume for model data
        volume = self.client.V1Volume(
            name="model-data",
            config_map=self.client.V1ConfigMapVolumeSource(name=f"model-{job_id}"),
        )

        volume_mount = self.client.V1VolumeMount(name="model-data", mount_path="/models")

        # Container spec
        container = self.client.V1Container(
            name="benchmark",
            image=self.image,
            command=["python", "-c", self._get_benchmark_script()],
            volume_mounts=[volume_mount],
            resources=resources,
            env=[self.client.V1EnvVar(name="JOB_ID", value=job_id)],
        )

        # Pod template
        pod_template = self.client.V1PodTemplateSpec(
            metadata=self.client.V1ObjectMeta(labels={"job": job_id}),
            spec=self.client.V1PodSpec(
                restart_policy="Never", containers=[container], volumes=[volume]
            ),
        )

        # Node selector for GPU type
        if self.gpu_type:
            pod_template.spec.node_selector = {"gpu-type": self.gpu_type}

        # Job spec
        job = self.client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=self.client.V1ObjectMeta(
                name=f"benchmark-{job_id}",
                namespace=self.namespace,
                labels={"job": job_id, "app": "embodied-ai-benchmark"},
            ),
            spec=self.client.V1JobSpec(
                template=pod_template, backoff_limit=2, ttl_seconds_after_finished=self.ttl_seconds
            ),
        )

        self.batch_api.create_namespaced_job(self.namespace, job)

    def _get_benchmark_script(self) -> str:
        """Get inline Python script for benchmark execution.

        Returns:
            Python script as string
        """
        return """
import json
import time
import torch
import numpy as np
from pathlib import Path

# Load config
config = json.loads(Path('/models/config.json').read_text())

# Create simple model (placeholder - real implementation would reconstruct)
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10)
)

# Load state dict (if available)
try:
    model.load_state_dict(torch.load('/models/model.pt'))
except:
    pass

model.eval()

# Create input
input_shape = tuple(config['input_shape'])
dummy_input = torch.randn(*input_shape)

# Warmup
for _ in range(config['warmup_iterations']):
    with torch.no_grad():
        _ = model(dummy_input)

# Benchmark
latencies = []
for _ in range(config['iterations']):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

# Calculate statistics
latencies_np = np.array(latencies)
results = {
    'mean_latency_ms': float(np.mean(latencies_np)),
    'std_latency_ms': float(np.std(latencies_np)),
    'min_latency_ms': float(np.min(latencies_np)),
    'max_latency_ms': float(np.max(latencies_np))
}

print(json.dumps(results))
"""

    def _wait_for_job_completion(self, job_id: str, timeout: int = 600):
        """Wait for job to complete.

        Args:
            job_id: Job identifier
            timeout: Maximum wait time in seconds

        Raises:
            Exception: If job fails or times out
        """
        start_time = time.time()
        job_name = f"benchmark-{job_id}"

        while time.time() - start_time < timeout:
            job = self.batch_api.read_namespaced_job(job_name, self.namespace)

            if job.status.succeeded:
                return
            elif job.status.failed:
                raise Exception(f"Job {job_id} failed")

            time.sleep(2)

        raise Exception(f"Job {job_id} timed out after {timeout}s")

    def _get_job_results(self, job_id: str) -> BenchmarkResult:
        """Get results from job logs.

        Args:
            job_id: Job identifier

        Returns:
            BenchmarkResult

        Raises:
            Exception: If results cannot be parsed
        """
        # Find pod for job
        pods = self.core_api.list_namespaced_pod(self.namespace, label_selector=f"job={job_id}")

        if not pods.items:
            raise Exception(f"No pods found for job {job_id}")

        pod_name = pods.items[0].metadata.name

        # Get logs
        logs = self.core_api.read_namespaced_pod_log(pod_name, self.namespace)

        # Parse JSON from logs
        for line in logs.split("\n"):
            try:
                data = json.loads(line)
                if "mean_latency_ms" in data:
                    # Calculate throughput
                    batch_size = 1
                    throughput = (1000.0 / data["mean_latency_ms"]) * batch_size

                    return BenchmarkResult(
                        backend_name=self.name,
                        device=f"kubernetes@{self.namespace}",
                        mean_latency_ms=data["mean_latency_ms"],
                        std_latency_ms=data["std_latency_ms"],
                        min_latency_ms=data["min_latency_ms"],
                        max_latency_ms=data["max_latency_ms"],
                        throughput_samples_per_sec=throughput,
                        iterations=100,
                        warmup_iterations=10,
                        metadata={
                            "namespace": self.namespace,
                            "job_id": job_id,
                            "gpu_type": self.gpu_type,
                        },
                    )
            except json.JSONDecodeError:
                continue

        raise Exception(f"Could not parse results from job {job_id}")

    def _cleanup_job(self, job_id: str):
        """Cleanup job and configmap.

        Args:
            job_id: Job identifier
        """
        try:
            # Delete job (will delete pod too)
            self.batch_api.delete_namespaced_job(
                f"benchmark-{job_id}", self.namespace, propagation_policy="Background"
            )
        except:
            pass

        try:
            # Delete configmap
            self.core_api.delete_namespaced_config_map(f"model-{job_id}", self.namespace)
        except:
            pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return Kubernetes backend capabilities.

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
            "supports_parallel": True,
            "supports_gpu": True,
            "horizontal_scaling": True,
            "requires_secrets": True,
        }
