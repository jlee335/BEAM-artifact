# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    context = "Hi " * 1
    context2 = "Hey " * 1
    return [
        context + "Hello, my name is",
        context + "The capital of France is",
        context2 + "Your name is",
        context2 + "The capital of China is",
    ]
model_name="ByteResearch/Llama-3-8B-Instruct"

storage_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": "local_storage"})

lmcache_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_producer",
            kv_connector_extra_config={"lmcache_path": "local_storage"})

def main():
    prompts = read_prompts()

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # no caching
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        kv_transfer_config=storage_config,
    )  # , max_model_len=2048, max_num_batched_tokens=2048)

    # 1ST generation (prefill instance)
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    new_prompts = []
    print("-" * 30)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 30)

    # Write new_prompts to output.txt
    with open("output.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(new_prompts)} prompts to output.txt")


if __name__ == "__main__":
    main()
