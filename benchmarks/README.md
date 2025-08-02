
# Benchmark Instructions

Welcome! 👋 You are currently in the `benchmarks/` directory. All commands should be run from this location unless specified otherwise. Before you get started, please make sure you have followed all instructions in the root `README.md` folder to install `pie`.

---

## System Requirements

* **GPU:** An NVIDIA GPU with at least **24GB of VRAM** is required for the default configuration.
* **Note for smaller GPUs:** You can still run the benchmarks on GPUs with less memory, but you may need to adjust workload sizes (e.g., smaller `--num-instances` for throughput measurement scripts) to avoid out-of-memory errors.

---

## Environment Setup

Please follow these steps in order to set up the required dependencies and components.

### Step 1: Install System Dependencies

You'll need Docker, the NVIDIA Container Toolkit.

1.  **Install Docker:** Follow the official installation guide for your OS.
2.  **Install NVIDIA Container Toolkit:** This allows Docker containers to access the GPU.
    * <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

### Step 2: Install Python Dependencies

Install the necessary Python packages for the evaluation scripts.

```bash
pip install -r requirements.txt
````

### Step 3: Download LLM Models

We pre-download models to ensure consistency and avoid issues with gated access from Hugging Face.

1.  **Install and Login to [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli):** You'll need an access token.
    ```bash
    huggingface-cli login
    ```
2.  **Download the models:**
    ```bash
    huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
    huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
    ```

### Step 4: Compile PIE Inferlets

The benchmark applications, or "inferlets," need to be compiled from their Rust source code.

```bash
cd ../example-apps
cargo build --target wasm32-wasip2 --release
cd ../benchmarks
```

The compiled `*.wasm` files will be located in `../example-apps/target/wasm32-wasip2/release/`.

### Step 5: Verify Setup ✅

Before running the full benchmark suite, let's verify that all components are working correctly.

1.  **Start the PIE system:** We'll use the default evaluation configuration.
    ```bash
    cd ../pie-cli
    pie start --config ./example_config.toml
    ```
2.  **Verify vLLM and SGLang Docker containers:** These scripts will pull the Docker images (this may take some time) and start the baseline servers.
    ```bash
    chmod +x run_vllm.sh run_sglang.sh
    ./run_vllm.sh
    ./run_sglang.sh
    ```
    If these commands run without error, your setup is likely correct\!

-----

## Overview of Included Inferlets

Inferlets are the core artifacts of our work, implementing various generation strategies as self-contained WASM modules. The table below lists all available inferlets in the `../example-apps/` directory.

| Inferlet Name | Evaluated | Source Code Path |
| :--- | :---: | :--- |
| **Agents** | | |
| ReAct | ✅ | `../example-apps/agent-react/src/lib.rs` |
| CodeAct | ✅ | `../example-apps/agent-codeact/src/lib.rs` |
| Swarm | ✅ | `../example-apps/agent-swarm/src/lib.rs` |
| **Structured Generation** | | |
| Tree of Thought (ToT) | ✅ | `../example-apps/tree-of-thought/src/lib.rs` |
| Recursion of Thought (RoT)| ✅ | `../example-apps/recursion-of-thought/src/lib.rs` |
| Graph of Thought (GoT) | ✅ | `../example-apps/graph-of-thought/src/lib.rs` |
| Skeleton of Thought (SkoT)| ✅ | `../example-apps/skeleton-of-thought/src/lib.rs` |
| **Decoding & Sampling** | | |
| Text Completion | ✅ | `../example-apps/text-completion/src/lib.rs` |
| EBNF-based Decoding | ✅ | `../example-apps/constrained-decoding/src/lib.rs` |
| Beam Search | ✅ | `../example-apps/beam-search/src/lib.rs` |
| N-gram Speculative Decoding| ✅ | `../example-apps/speculative-decoding/src/lib.rs` |
| Jacobi Decoding | | `../example-apps/jacobi-decoding/src/lib.rs` |
| **System Optimizations** | | |
| Prefix Caching | ✅ | `../example-apps/prefix-caching/src/lib.rs` |
| Attention Sink | ✅ | `../example-apps/attention-sink/src/lib.rs` |
| Windowed Attention | | `../example-apps/windowed-attention/src/lib.rs` |
| Hierarchical Attention | | `../example-apps/hierarchical-attention/src/lib.rs` |
| **Other Applications** | | |
| Watermarking | | `../example-apps/watermarking/src/lib.rs` |
| Output Validation | | `../example-apps/output-validation/src/lib.rs` |

-----

## Running the Benchmarks

This section details how to run the scripts to reproduce the results from the paper.

**General Instructions:**

* **PIE vs. Baselines:** The benchmark scripts are separated. Scripts named `*_pie.py` are for evaluating our system, **PIE**. Scripts named `*_baseline.py` or `*_sglang.py` are for evaluating the **vLLM/SGLang** baselines.
    * For `*_pie.py` scripts, ensure the PIE system is running (`pie start ...`).
    * For baseline scripts, ensure the appropriate baseline server is running (e.g., `./run_vllm.sh`).
* **Configuration:** PIE's behavior (e.g., model, batching strategy) is controlled by `../pie-cli/example_config.toml`. You will need to edit this file to reproduce certain results.
* **Throughput vs. Latency:** By default, scripts measure **throughput**. To measure **latency**, run with a single request: `--num-requests 1`.
* **Results:** All results are saved as `.log` files in the `logs/` directory.

### Quick Start: Sanity Check 🚀

To quickly confirm that the whole pipeline is working, run a simple text completion benchmark for PIE.

```bash
# Make sure PIE is running in another terminal
python test_5_text_completion_pie.py
```

This should complete without errors with an output something like this, and generate `logs/test_5_text_completion_pie.py.log`.

```
# --- ✅ Benchmark Complete ---
# Total Time Taken:       1217.71 milliseconds
# Throughput:             105.12 requests/second
# --------------------------
```
Note that the very first run may take longer.

### Reproducing Paper Results

#### Figures 6 & 7: Agent Benchmarks

* **ReAct Agent**
    * PIE: `python test_1_agent_react_pie.py`
    * Baseline: `python test_1_agent_react_baseline.py`
* **CodeAct Agent**
    * PIE: `python test_2_agent_codeact_pie.py`
    * Baseline: `python test_2_agent_codeact_baseline.py`
* **Swarm Agent**
    * PIE: `python test_3_agent_swarm_pie.py`
    * Baseline: `python test_3_agent_swarm_baseline.py`
* **Figure 7 (Case Study: ReAct Optimization)**
  ```bash
  # This script automates running the PIE experiments with different optimizations
  ./test_4_agent_case_study_pie.sh
  # This script generates the plot from the results
  python test_4_agent_case_study_pie_viz.py
  ```
  **Note:** For demonstration, this shell script only runs each experiment once. Please repeat the experiment manually if you want to reduce noise.

#### Figure 8: Task Benchmarks

For each task, run the `_pie.py` script against a running PIE instance and the `_baseline.py` or `_sglang.py` script against a running vLLM/SGLang instance.

* **Text Completion**
    * PIE: `python test_5_text_completion_pie.py`
    * Baseline: `python test_5_text_completion_baseline.py`
* **PrefixTree**
    * PIE: `python test_6_prefix_tree_pie.py`
    * vLLM: `python test_6_prefix_tree_baseline.py`
    * SGLang: `python test_6_prefix_tree_sglang.py`
* **Tree of Thought (ToT)**
    * PIE: `python test_7_tot_pie.py`
    * vLLM: `python test_7_tot_baseline.py`
    * SGLang: `python test_7_tot_sglang.py`
* **Recursion of Thought (RoT)**
    * PIE: `python test_8_rot_pie.py`
    * Baseline: `python test_8_rot_baseline.py`
* **Graph of Thought (GoT)**
    * PIE: `python test_9_got_pie.py`
    * Baseline: `python test_9_got_baseline.py`
* **Skeleton of Thought (SkoT)**
    * PIE: `python test_10_skot_pie.py`
    * Baseline: `python test_10_skot_baseline.py`
* **Prompt Caching**
    * PIE: `python test_11_cache_pie.py`
    * Baseline: `python test_11_cache_baseline.py` (Note: Use `./run_vllm_apc.sh` to start vLLM with caching enabled).
* **EBNF-based Constrained decoding**
    * PIE: `python test_12_ebnf_pie.py`
    * Baseline: `python test_12_ebnf_baseline.py` (Note: Use `./run_vllm_ebnf.sh` to start vLLM with EBNF enabled).
* **N-gram based Speculative Decoding**
    * PIE: `python test_13_specdec_pie.py`
    * Baseline: `python test_13_specdec_baseline.py` (Note: Use `./run_vllm_specdec.sh` to start vLLM with speculative decoding).
* **Beam Search**
    * PIE: `python test_14_beamsearch_pie.py`
    * Baseline: `python test_14_beamsearch_baseline.py`
* **Attention Sink**
    * PIE: `python test_15_attnsink_pie.py`

#### Figures 9, 11 & Tables 4, 5: Microbenchmarks

* **Figure 9 (Inferlet Spawn Time):**
  ```bash
  ./microbench_spawn_time.sh
  python microbench_spawn_time_viz.py
  ```
* **Figure 11 (API Call Overhead):**
  ```bash
  ./microbench_execution_latency.sh
  python microbench_execution_latency_viz.py
  ```
* **Table 4 (Time-Per-Output-Token vs. Model Size):**
    1.  Modify `../pie-cli/example_config.toml` to select the desired model.
        ```toml
        [[backend]]
        # Options: "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"
        model = "meta-llama/Llama-3.2-1B-Instruct"
        ```
    2.  Restart PIE to apply the config change.
    3.  Run the text completion benchmark: `python test_5_text_completion_pie.py`. The TPOT will be in the log file.
* **Table 5 (Batching Strategy Throughput):**
    1.  Modify `../pie-cli/example_config.toml` to select the batching strategy.
        ```toml
        # Options: "adaptive", "k", "t", "kort" (default: "adaptive")
        batching_strategy = "adaptive"
        batching_strategy_k = 8
        batching_strategy_t = 16 # in milliseconds
        ```
    2.  Restart PIE to apply the config change.
    3.  Run the text completion benchmark: `python test_5_text_completion_pie.py`.

-----

### Interactive Exploration (Optional)

You can also run inferlets interactively using the `pie-cli` shell for debugging or experimentation. After starting PIE, you can use the `run` command in the shell:

```bash
pie> run ../example-apps/target/wasm32-wasip2/release/text_completion.wasm -- --prompt "It is Friday afternoon in Seattle, what should I do this weekend?"

✅ Inferlet launched with ID: ...
[Inst ...] Output: "Since it's a beautiful Friday in Seattle, you're in for a great weekend! You could start by visiting the Fremont Troll under the bridge, then grabbing a coffee at a local cafe..."
...
```

This provides a flexible way to test modifications or your own custom inferlets. 