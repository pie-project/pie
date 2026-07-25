import json, os, sys, time
sys.path.insert(0, 'src')
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from gpu_lr1.vllm_backend import install
if os.environ.get("BACKEND", "gpugrammar") == "gpugrammar":
    install()
    print(">>> using gpugrammar backend")
else:
    print(">>> using stock xgrammar backend")

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}

llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=1024, gpu_memory_utilization=0.35,
          enforce_eager=False, structured_outputs_config={"backend": "xgrammar"})
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=96,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
prompts = [f"Give a JSON profile for person {i}. JSON only." for i in range(16)]

t0 = time.perf_counter()
outputs = llm.generate(prompts, params)
dt = time.perf_counter() - t0

valid = 0
for output in outputs:
    text = output.outputs[0].text.strip()
    try:
        value = json.loads(text)
        assert set(value) == {"name", "age", "active"}
        assert isinstance(value["age"], int) and isinstance(value["active"], bool)
        valid += 1
    except Exception as error:
        print("INVALID:", repr(text[:120]), error)
tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"{valid}/{len(outputs)} valid | {tokens} tokens in {dt:.2f}s = {tokens/dt:.0f} tok/s")
