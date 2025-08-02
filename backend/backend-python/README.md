
# PIE Torch Backend


### Step 1: Install the FlashInfer Library


1.  **Determine your GPU's CUDA Compute Capability.**
    Run the following command to find your GPU's compute capability version. You will need this for the next step.

    ```bash
    nvidia-smi --query-gpu=compute_cap --format=csv
    ```

    This will output a version number, for example, `8.6`.

2.  **Clone, Configure, and Build FlashInfer.**
    The following script clones the FlashInfer repository and builds it. **Important:** Before running, replace `"8.6"` in the `export TORCH_CUDA_ARCH_LIST` line with the compute capability you found in the previous step.

    ```bash
    pip install torch # Ensure you have the PyTorch installed
    
    # Clone the repository and its submodules
    git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
    cd flashinfer

    # IMPORTANT: Set your target CUDA architecture below
    export TORCH_CUDA_ARCH_LIST="8.6" # Replace "8.6" with your compute_cap value

    # Build the Ahead-of-Time (AOT) kernels. This will take some time (~5-10 minutes).
    python -m flashinfer.aot

    # Build the wheel package
    python -m pip install --no-build-isolation --verbose .

    cd ../
    ```

### Step 2: Compile Protobuf Definitions

This step uses the Protocol Buffers compiler (`protoc`) to generate the necessary Python code for data serialization and communication within the backend.

```bash
# Make the script executable
chmod +x build_proto.sh

# Run the script
./build_proto.sh
````

### Step 3: Install the PIE Torch Backend

Now, you can install the backend package itself. We use an "editable" install (`-e`), which is useful for development as it allows you to make changes to the source code without needing to reinstall the package.

```bash
pip install -e .
```

Once these steps are complete, the PIE Torch backend will be successfully installed.
