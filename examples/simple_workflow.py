"""Simple example demonstrating the Embodied AI Architect workflow."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from embodied_ai_architect import Orchestrator
from embodied_ai_architect.agents import ModelAnalyzerAgent


# Define a simple CNN model for image classification
class SimpleCNN(nn.Module):
    """Simple CNN for demonstration purposes."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    """Run a simple workflow example."""

    print("=" * 60)
    print("Embodied AI Architect - Simple Workflow Example")
    print("=" * 60)

    # Create a simple CNN model
    print("\n📦 Creating a simple CNN model...")
    model = SimpleCNN(num_classes=10)
    print(f"✓ Model created: {type(model).__name__}")

    # Initialize orchestrator
    print("\n🎯 Initializing Orchestrator...")
    orchestrator = Orchestrator()

    # Register agents
    print("\n🤖 Registering agents...")
    model_analyzer = ModelAnalyzerAgent()
    orchestrator.register_agent(model_analyzer)

    print(f"\n📋 Registered agents: {orchestrator.list_agents()}")

    # Create request
    request = {
        "model": model,
        "input_shape": (1, 3, 32, 32),  # Batch size 1, 3 channels, 32x32 image
        "targets": ["cpu", "gpu"],
        "constraints": {
            "max_latency_ms": 100,
            "max_power_watts": 15
        }
    }

    # Process the request
    result = orchestrator.process(request)

    # Display results
    print("\n📊 Detailed Results:")
    print("-" * 60)

    if result.status.value == "completed":
        print("\nWorkflow Summary:")
        print(f"  Status: {result.status.value}")
        print(f"  Total Agents Executed: {result.summary.get('total_agents_executed')}")
        print(f"  Successful: {result.summary.get('successful_agents')}")
        print(f"  Failed: {result.summary.get('failed_agents')}")

        if "model_summary" in result.summary:
            model_sum = result.summary["model_summary"]
            print(f"\n  Quick Stats:")
            print(f"    Parameters: {model_sum.get('total_parameters'):,}")
            print(f"    Layers: {model_sum.get('total_layers')}")

    else:
        print(f"\n❌ Workflow failed: {result.error}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
