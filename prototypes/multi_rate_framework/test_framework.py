"""Quick validation test for the framework.

This test validates core concepts without needing Zenoh.
"""

import time
from framework import Component, Input, Output, control_loop, RateConfig


def test_decorator():
    """Test that @control_loop decorator works."""
    print("Testing @control_loop decorator...")

    @control_loop(rate_hz=100, priority=50, deadline_ms=5.0)
    class TestComponent(Component):
        def step(self, dt):
            pass

    # Check rate config was attached
    assert hasattr(TestComponent, '__rate_config__')
    assert TestComponent.__rate_config__.rate_hz == 100
    assert TestComponent.__rate_config__.priority == 50
    assert TestComponent.__rate_config__.deadline_ms == 5.0
    assert TestComponent.__rate_config__.period_sec == 0.01  # 1/100

    print("  ✅ Decorator works correctly")


def test_input_output():
    """Test Input/Output descriptors."""
    print("Testing Input/Output descriptors...")

    @control_loop(rate_hz=10)
    class TestComponent(Component):
        input_value: Input[int] = Input("test/input", default=42)
        output_value: Output[str] = Output("test/output")

        def step(self, dt):
            pass

    # Check that inputs/outputs were registered
    assert hasattr(TestComponent, '__inputs__')
    assert hasattr(TestComponent, '__outputs__')
    assert 'input_value' in TestComponent.__inputs__
    assert 'output_value' in TestComponent.__outputs__

    # Check topic names
    assert TestComponent.__inputs__['input_value'].topic == "test/input"
    assert TestComponent.__outputs__['output_value'].topic == "test/output"

    # Test instance behavior
    comp = TestComponent()
    assert comp.input_value == 42  # Default value

    # Test output assignment
    TestComponent.output_value.__set__(comp, "hello")
    assert TestComponent.output_value.__get__(comp, TestComponent) == "hello"

    print("  ✅ Input/Output descriptors work correctly")


def test_component_lifecycle():
    """Test component lifecycle methods."""
    print("Testing component lifecycle...")

    lifecycle_log = []

    @control_loop(rate_hz=10)
    class TestComponent(Component):
        def on_start(self):
            lifecycle_log.append("start")

        def step(self, dt):
            lifecycle_log.append("step")

        def on_stop(self):
            lifecycle_log.append("stop")

    comp = TestComponent()
    comp.on_start()
    comp.step(0.1)
    comp.step(0.1)
    comp.on_stop()

    assert lifecycle_log == ["start", "step", "step", "stop"]

    print("  ✅ Lifecycle methods work correctly")


def test_timing():
    """Test basic timing with manual scheduler simulation."""
    print("Testing timing accuracy...")

    @control_loop(rate_hz=100)  # 10ms period
    class TestComponent(Component):
        def step(self, dt):
            self.tick_count += 1

    comp = TestComponent()

    # Simulate 10 ticks at 100 Hz
    start = time.monotonic()
    period = 1.0 / 100  # 10ms

    next_tick = time.monotonic()
    for i in range(10):
        comp.step(period)
        next_tick += period
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

    elapsed = time.monotonic() - start
    expected = 10 * period  # Should be ~100ms

    # Allow 10% tolerance
    assert abs(elapsed - expected) / expected < 0.1, \
        f"Timing error: {elapsed:.3f}s vs {expected:.3f}s"

    print(f"  ✅ Timing accurate: {elapsed*1000:.1f}ms (expected {expected*1000:.1f}ms)")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Framework Validation Tests")
    print("=" * 70)
    print()

    try:
        test_decorator()
        test_input_output()
        test_component_lifecycle()
        test_timing()

        print()
        print("=" * 70)
        print("✅ All validation tests passed!")
        print("=" * 70)
        print()
        print("Framework core concepts are working correctly.")
        print("Ready to test with Zenoh communication.")
        return 0

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ Test failed: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Unexpected error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
