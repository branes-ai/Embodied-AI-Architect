"""Simple two-component example demonstrating multi-rate control.

This example shows:
- FastProducer running at 100 Hz
- SlowConsumer running at 1 Hz
- Communication via Zenoh
- SlowConsumer gets latest value from FastProducer
"""

from framework import Component, Application, Input, Output, control_loop


@control_loop(rate_hz=100)
class FastProducer(Component):
    """Fast component publishing at 100 Hz."""

    # Output
    count: Output[int] = Output("demo/count")

    def __init__(self):
        super().__init__()
        self.counter = 0

    def step(self, dt):
        """Called at 100 Hz."""
        self.count = self.counter
        self.counter += 1

        # Log every 100 ticks (once per second)
        if self.tick_count % 100 == 0:
            print(f"[FastProducer @ 100 Hz] Published count={self.counter}")


@control_loop(rate_hz=1)
class SlowConsumer(Component):
    """Slow component consuming at 1 Hz."""

    # Input
    count: Input[int] = Input("demo/count", default=0)

    def __init__(self):
        super().__init__()
        self.last_count = 0

    def step(self, dt):
        """Called at 1 Hz."""
        current_count = self.count
        updates_received = current_count - self.last_count

        print(
            f"[SlowConsumer @ 1 Hz] Received count={current_count} "
            f"(got {updates_received} updates in {dt:.1f} seconds)"
        )

        self.last_count = current_count


def main():
    """Run the simple example."""

    print("=" * 70)
    print("Simple Multi-Rate Example")
    print("=" * 70)
    print("FastProducer: 100 Hz")
    print("SlowConsumer: 1 Hz")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Create application
    app = Application(name="simple_multirate")

    # Register components
    app.register(FastProducer())
    app.register(SlowConsumer())

    # Run (blocks until Ctrl+C)
    app.run()


if __name__ == "__main__":
    main()
