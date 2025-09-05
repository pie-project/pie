# PIE CUDA Backend


**Required:**

  * CMake (version 3.18 or higher).
  * Ninja build system.
  * NVIDIA CUDA Toolkit (version 11.x or 12.x is recommended). Make sure `nvcc` is in your system's `PATH`.
  * A C++20 compliant compiler
  * A NVIDIA GPU with SM 8.0 or higher (e.g., RTX 30 series, RTX 40 series, or A100).

**Library Dependencies:**

  * ZeroMQ (`libzmq3-dev`): Required for communication with the PIE engine.
  * CBOR (`libcbor-dev`): Required for zTensor
  * Zstandard (`libzstd-dev`): Required for zTensor

On Debian, you can install these with:

```bash
sudo apt-get update && sudo apt-get install -y \
  git \
  cmake \
  ninja-build \
  libzmq3-dev \
  libcbor-dev \
  libzstd-dev
```



## Building from Source


These commands will create a `build` directory, configure the project with CMake, and compile it with Ninja.

```bash
mkdir -p build && cd build
cmake .. -G Ninja
ninja
```

## Running Unit Tests

Build and execute all CUDA unit test binaries (they live in `build/bin/*`):

```bash
make run-unit-tests
```

This configures (if needed), builds the test executables, and runs them sequentially, failing fast on the first error.


## Running the Backend


```bash
./bin/pie_cuda_be --config dev.toml
```

You can customize the server's behavior by editing the `.toml` file.
