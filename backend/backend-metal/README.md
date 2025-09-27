# PIE Metal Backend

Metal-based computation backend for PIE inference acceleration.

## Build Requirements

- macOS with Metal support
- CMake 3.23+
- Xcode command line tools
- Python 3.11+ with pybind11

## Building

```bash
# Configure
cmake -B build -S .

# Build Metal kernels and Python bindings
cmake --build build

# Test bindings
cd build/lib
python3 -c "import metal_bindings; print('Success!')"
```

## Output Files

- `build/lib/pie_metal_kernels.metallib` - Compiled Metal kernels
- `build/lib/metal_bindings.cpython-*.so` - Python bindings module

## Usage

The Metal backend provides accelerated inference for L4MA models:

```python
from metal_backend import MetalBackend
from l4ma_runtime import MetalL4maBackend

# Initialize Metal backend
metal_backend = MetalBackend(model_metadata={"architecture": arch_dict})
if metal_backend.initialize():
    runtime = MetalL4maBackend(metal_backend=metal_backend)
    # Use with your L4MA model
```

