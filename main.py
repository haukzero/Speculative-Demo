import utils
import sampling
from time import perf_counter_ns
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ZOO = [
    "/root/models/opt-125m",
    "/root/models/opt-350m",
    "/root/models/opt-1.3b",
    "/root/models/opt-2.7b",
    "/root/models/opt-6.7b",
]
approx_model = MODEL_ZOO[ 2 ]
target_model = MODEL_ZOO[ 3 ]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(approx_model)
    approx_model = AutoModelForCausalLM.from_pretrained(approx_model, device_map="auto")
    target_model = AutoModelForCausalLM.from_pretrained(target_model, device_map="auto")
    prompt = "What is love? Everyone has different answers."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(approx_model.device)

    max_length = 128
    top_k = 32
    top_p = 0.9
    temperature = 1.0
    gamma = 4
    random_seed = 666

    start = perf_counter_ns() / 1e6
    autoreg_output = sampling.autoregressive_sample(
        input_ids,
        target_model,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    end = perf_counter_ns() / 1e6
    utils.green_print(f"Autoregressive sampling time: {end - start} ms")
    print("Decoded output:")
    utils.blue_print(tokenizer.decode(autoreg_output[ 0 ], skip_special_tokens=True))
    print("=" * 100)

    start = perf_counter_ns() / 1e6
    spec_output = sampling.speculative_sampling(
        input_ids,
        approx_model,
        target_model,
        max_length=max_length,
        gamma=gamma,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        random_seed=random_seed,
    )
    end = perf_counter_ns() / 1e6
    utils.green_print(f"Speculative sampling time: {end - start} ms")
    print("Decoded output:")
    utils.blue_print(tokenizer.decode(spec_output[ 0 ], skip_special_tokens=True))
