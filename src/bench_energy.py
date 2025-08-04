import random
import time
import os

import pandas as pd
import torch
import json

from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from zeus.monitor import ZeusMonitor


# TODO: Consider adding more parameters to this function
def create_warmup_measure_dataset(tokenizer):
    ds = load_dataset("agentlans/high-quality-english-sentences", split="test")

    token_lengths = [len(tokenizer.encode(x["text"], add_special_tokens=False)) for x in ds]

    sample_per_bucket = 4
    warmup_samples = 2

    buckets = {
        "A": {"min": 0, "max": 15, "indices": []},
        "B": {"min": 16, "max": 30, "indices": []},
        "C": {"min": 31, "max": 50, "indices": []},
        "D": {"min": 51, "max": 100, "indices": []},
    }

    for idx, length in enumerate(token_lengths):
        for b in buckets.values():
            if b["min"] <= length <= b["max"]:
                b["indices"].append(idx)
                break

    random.seed(42)
    final_indices = []
    for name, b in buckets.items():
        if name == "A":
            # bucket samples and warmup samples are drawn together to make sure every value is unique
            final_indices.extend(random.sample(b["indices"], sample_per_bucket + warmup_samples))
        else:
            final_indices.extend(random.sample(b["indices"], sample_per_bucket))

    warmup_indices = final_indices[:warmup_samples]
    final_indices = final_indices[warmup_samples:]

    warmup_indices = sorted(warmup_indices, key=lambda i: token_lengths[i])
    final_indices = sorted(final_indices, key=lambda i: token_lengths[i])

    return ds.select(warmup_indices), ds.select(final_indices)


# TODO: Add function comments
def bench(args):
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=args.device)
    warmup_sample, measurement_sample = create_warmup_measure_dataset(tokenizer)

    monitor = ZeusMonitor()
    base_out_path = os.path.join(args.path if args.path else os.getcwd(), model_name.replace('/', '_'))

    if args.warmup:
        prompts = warmup_sample["text"]
        for i in range(0, len(prompts), args.batch_size[0]):
            batch_prompts = prompts[i:i + args.batch_size[0]]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            model.generate(**inputs, do_sample=True, temperature=1, max_new_tokens=args.max_new_token,
                           pad_token_id=tokenizer.eos_token_id)

    prompts = measurement_sample["text"]

    torch.manual_seed(42)
    with torch.no_grad():
        for batch_size in args.batch_size:
            ins_outs = []
            measurements = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

                # accessing zeus timestamps is troublesome
                begin = time.time()
                monitor.begin_window("generate")
                outputs = model.generate(**inputs, do_sample=True, temperature=1, max_new_tokens=args.max_new_token,
                                         pad_token_id=tokenizer.eos_token_id)
                measurement = monitor.end_window("generate")
                end = time.time()

                for prompt, completion in zip(batch_prompts, outputs):
                    ins_outs.append({"prompt": prompt,
                                     "completion": tokenizer.
                                    decode(completion, skip_special_tokens=True)[len(prompt):]})

                # the number of batched prompts may be smaller than the batch size, so multiply by the length of current batch prompt length
                measurements.append({"measurement": measurement, "start": begin, "end": end,
                                     "generated_tokens": len(outputs[0][len(inputs[0]):]) * len(batch_prompts)})

            df = create_dataframe(measurements)
            save_results(df, ins_outs,
                         os.path.join(base_out_path, f"batch-size{batch_size}_token{args.max_new_token}"))


def create_dataframe(measurements: list[dict]) -> DataFrame:
    df = pd.DataFrame(columns=['time(s)', 'cpu_energy(J)', 'gpu_energy(J)', 'start', 'end', "generated_tokens"])

    for _measurement in measurements:
        measurement = _measurement['measurement']
        start = _measurement['start']
        end = _measurement['end']
        generated_tokens = _measurement['generated_tokens']
        if not measurement.cpu_energy:
            # If Zeus can't find any CPU, the cpu_energy dict is none, so here it's manually set to 0
            measurement.cpu_energy = {0: 0}
        df.loc[len(df)] = {
            'time(s)': measurement.time,
            'cpu_energy(J)': sum(
                measurement.cpu_energy.values()),
            'gpu_energy(J)': sum(
                measurement.gpu_energy.values()),
            'start': start,
            'end': end,
            'generated_tokens': generated_tokens
        }

    return df


def save_results(df: DataFrame, ins_outs: list[dict[str, str]], out_path: str):
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, "ins_outs.json"), "w", encoding="utf8") as file:
        json.dump(ins_outs, file, indent=4)

    df.to_csv(os.path.join(out_path, "energy.csv"), sep=";", index=False, mode="w", header=True)
