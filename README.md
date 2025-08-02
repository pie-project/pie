# PIE: A Programmable Serving System for Emerging LLM Applications

This repository contains the artifact for the SOSP 2025 paper, "PIE: A Programmable Serving System for Emerging LLM Applications."

PIE is a novel serving system for LLMs that enables users to write and deploy *inferlets*—lightweight, user-defined programs that control the LLM inference process from end to end.

-----

## Repository Structure

This artifact is composed of several key components:

* `pie`: The main implementation of PIE's application and control layers.
* `pie-cli`: A command-line interface for interacting with PIE.
* `backend`: PIE's inference layer, implemented in Python.
* `inferlet`: A Rust crate to simplify writing new inferlets.
* `example-apps`: A collection of example inferlets used in our evaluation.
* `benchmarks`: Scripts to reproduce the performance benchmarks from the paper.
* `client`: A Python client library for programmatic interaction with PIE.

-----

## Prerequisites

Before you begin, please ensure your system meets the following requirements:

* **GPU:** NVIDIA GPU (Ampere architecture or newer).
* **CUDA:** CUDA Toolkit `12.6` or later.
* **OS:** Linux (Ubuntu 22.04 or later is recommended).
* **Rust:** [Rust Toolchain](https://www.rust-lang.org/tools/install) `1.80` or later.
* **Python:** [Python](https://www.python.org/downloads/) `3.11` or later.

-----

## Installation & Setup

Follow these steps to install all components of the PIE system.

### Step 1: Install Core Python Components (Backend & Client)

The PIE engine relies on a Python-based backend for inference.

1.  **Install PIE Torch Backend:** Follow the instructions in [backend/README.md](./backend/backend-python/README.md) to set up the inference engine.
2.  **Install PIE Python Client:** Follow the instructions in [client/python/README.md](client/python/README.md) to install the client library.

### Step 2: Install the PIE Command-Line Interface (CLI)

The `pie-cli` is the primary tool for managing the PIE system.
If you don't have Rust installed, please follow the [Rust installation guide](https://www.rust-lang.org/tools/install).

```bash
cd pie-cli
cargo install --path .
cd ..
```

Verify the installation by checking the help message:

```bash
pie --help
```

### Step 3: Download and Register LLMs

Next, download the models used in our examples and benchmarks.
Download all the models will quite a bit of time and ~30 GB of disk space.

```bash
pie model add "llama-3.2-1b-instruct"
pie model add "llama-3.2-3b-instruct"
pie model add "llama-3.1-8b-instruct"
```

You can list all registered models to confirm they were added correctly:

```bash
pie model list
```

### Step 4: Compile Example Inferlets

The example applications must be compiled from Rust into WebAssembly.

1.  **Add the `wasm32-wasip2` target to your Rust toolchain:**
    ```bash
    rustup target add wasm32-wasip2
    ```
2.  **Compile the inferlets:**
    ```bash
    cd example-apps
    cargo build --target wasm32-wasip2 --release
    cd ..
    ```

The compiled `*.wasm` files will be located in the `example-apps/target/wasm32-wasip2/release/` directory.

-----

## Sanity Check

After completing the setup, you can run this simple check to ensure everything is working correctly.

1.  **Start the PIE Engine** Launch the engine with the example configuration file. This will start the backend services and open the interactive PIE shell.

    ```bash
    cd pie-cli
    pie start --config example_config.toml
    ```

    Wait for the confirmation message before proceeding:

    ```
    INFO pie::model: Backend service started
    ```

2.  **Run an Inferlet** From within the PIE shell (`pie>`), run the `text_completion.wasm` inferlet with a sample prompt.

    ```bash
    pie> run ../example-apps/target/wasm32-wasip2/release/text_completion.wasm -- --prompt "What is the capital of France?" --max-tokens 256
    ```

    If the setup is correct, you will see the inferlet launch and produce output, similar to this:

    ```
    ✅ Inferlet launched with ID: 812ad8ac-bbed-4e22-ab13-3dd77c8aa122
    pie> [Inst 812ad8ac] Output: "The capital of France is Paris." (total elapsed: 60.022073ms)
    [Inst 812ad8ac] Per token latency: 7.506236ms
    [Inferlet 812ad8ac-bbed-4e22-ab13-3dd77c8aa122] Terminated. Reason: instance normally finished
    ```

-----

## Next Steps: Running the Benchmarks

Now that the system is installed and verified, you can proceed to reproduce the performance results from our paper.

Please visit [`benchmarks/README.md`](benchmarks/README.md) for detailed instructions.