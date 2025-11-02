# Multi-Rate Control Framework Prototype

This prototype validates the core architecture for multi-rate control using Zenoh.

## What This Prototype Demonstrates

1. **Multi-rate scheduling**: Components running at different frequencies
2. **Zenoh pub/sub**: Automatic communication between components
3. **Declarative components**: Using decorators and type annotations
4. **Real timing**: Actual thread-based execution with timing validation

## Architecture Validation

This minimal prototype validates:
- ✅ Component model with `@control_loop` decorator
- ✅ Input/Output descriptors for automatic Zenoh wiring
- ✅ Multi-threaded scheduler (one thread per component)
- ✅ Async data caching (Zenoh callbacks)
- ✅ Different execution rates (1 Hz, 10 Hz, 100 Hz)
- ✅ Time-triggered execution with monotonic clock

## Installation

```bash
# Install Zenoh Python
pip install eclipse-zenoh

# Or from source
pip install git+https://github.com/eclipse-zenoh/zenoh-python
```

## Files

- `framework.py`: Minimal framework implementation
- `example_simple.py`: Simple two-component example
- `example_multirate.py`: Three-component example at different rates
- `run_prototype.sh`: Run all examples

## Running

```bash
# Simple example (fast producer, slow consumer)
python example_simple.py

# Multi-rate example (1 Hz, 10 Hz, 100 Hz)
python example_multirate.py
```

## Expected Output

```
[FastProducer @ 100 Hz] Publishing count=0
[SlowConsumer @ 1 Hz] Received count=100 (got 100 updates in 1 second)
[FastProducer @ 100 Hz] Publishing count=100
[FastProducer @ 100 Hz] Publishing count=200
[SlowConsumer @ 1 Hz] Received count=200 (got 100 updates in 1 second)
```

This demonstrates:
- Fast component runs at 100 Hz
- Slow component runs at 1 Hz
- Slow component receives latest value from fast component
- No blocking between components

## Validation Criteria

✅ **Pass**: Components run at specified rates (within 5% jitter)
✅ **Pass**: Components don't block each other
✅ **Pass**: Latest data is always available
✅ **Pass**: Clean shutdown on Ctrl+C

## Next Steps

If prototype validates successfully:
1. Expand to full framework implementation
2. Add deadline monitoring
3. Add CPU affinity and priority
4. Add proper serialization (msgpack/protobuf)
5. Add Rust components for ultra-fast loops
