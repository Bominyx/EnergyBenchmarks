import random
import time
import os

import pandas as pd
import torch
import json

from datasets import load_dataset, DatasetDict, load_from_disk
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM
from zeus.monitor import ZeusMonitor
from zeus.utils.logging import get_logger

# re-using logger from zeus to keep console log uniform
logger = get_logger(name=__name__)


def create_warmup_measure_dataset(tokenizer, hf_dataset_path="agentlans/high-quality-english-sentences", seed=42):
    ds = load_dataset(hf_dataset_path, split="test")

    token_lengths = [len(tokenizer.encode(x["text"], add_special_tokens=False)) for x in ds]

    sample_per_bucket = 250
    warmup_samples = 20

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

    random.seed(seed)
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

    warmup_dataset = ds.select(warmup_indices)
    measurement_dataset = ds.select(final_indices)

    dataset_dict = DatasetDict({
        "warmup": warmup_dataset,
        "measurement": measurement_dataset
    })

    return dataset_dict


# TODO: Add function comments
def bench(args):
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=args.device)

    if args.dataset_path:
        logger.info("Loading benchmark dataset from disk")
        dataset = load_from_disk(args.dataset_path)
    else:
        logger.info("Creating benchmark dataset in-memory")
        dataset = create_warmup_measure_dataset(tokenizer)

    monitor = ZeusMonitor()
    base_out_path = os.path.join(args.path if args.path else os.getcwd(), model_name.replace('/', '_'))

    if args.warmup:
        logger.info("Starting warmup")
        prompts = dataset["warmup"]["text"]
        for i in range(0, len(prompts), args.batch_size[0]):
            batch_prompts = prompts[i:i + args.batch_size[0]]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            model.generate(**inputs, do_sample=True, temperature=1, max_new_tokens=args.max_new_token,
                           pad_token_id=tokenizer.eos_token_id)
        logger.info("Warmup finished")

    prompts = dataset["measurement"]["text"]

    torch.manual_seed(42)
    with torch.no_grad():
        for batch_size in args.batch_size:
            logger.info(f"Benchmarking with batch size {batch_size}")
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
                                     "tokens": len(outputs[0][len(inputs[0]):]) * len(batch_prompts)})

            logger.info(f"Benchmark with batch size {batch_size} finished")
            run_folder = f"batch-size{batch_size}_token{args.max_new_token}"
            df_verbose, df_summary = create_dataframe(measurements, batch_size, run_folder)
            save_results(df_verbose, df_summary, ins_outs, os.path.join(base_out_path, run_folder))


def create_dataframe(measurements: list[dict], batch_size: int, run_folder: str) -> tuple[DataFrame, DataFrame]:
    df_verbose = pd.DataFrame(columns=['time(s)', 'cpu_energy(J)', 'gpu_energy(J)', 'start', 'end', "tokens"])

    for _measurement in measurements:
        measurement = _measurement['measurement']
        start = _measurement['start']
        end = _measurement['end']
        tokens = _measurement['tokens']
        if not measurement.cpu_energy:
            # If Zeus can't find any CPU, the cpu_energy dict is none, so here it's manually set to 0
            measurement.cpu_energy = {0: 0}
        df_verbose.loc[len(df_verbose)] = {
            'time(s)': measurement.time,
            'cpu_energy(J)': sum(
                measurement.cpu_energy.values()),
            'gpu_energy(J)': sum(
                measurement.gpu_energy.values()),
            'start': start,
            'end': end,
            'tokens': tokens
        }

    total_time = df_verbose['time(s)'].sum()
    total_cpu_energy = df_verbose['cpu_energy(J)'].sum()
    total_gpu_energy = df_verbose['gpu_energy(J)'].sum()
    avg_time = df_verbose['time(s)'].mean()
    avg_cpu_energy = df_verbose['cpu_energy(J)'].mean()
    avg_gpu_energy = df_verbose['gpu_energy(J)'].mean()
    total_tokens = df_verbose['tokens'].sum()
    avg_tokens = df_verbose['tokens'].mean()

    df_summary = pd.DataFrame({
        'run': run_folder,
        'total_time(s)': [total_time],
        'avg_time(s)': [avg_time],
        'total_cpu_energy(J)': [total_cpu_energy],
        'avg_cpu_energy(J)': [avg_cpu_energy],
        'total_gpu_energy(J)': [total_gpu_energy],
        'avg_gpu_energy(J)': [avg_gpu_energy],
        'total_tokens': [total_tokens],
        'avg_tokens': [avg_tokens],
        'batch_size': [batch_size],
    })
    return df_verbose, df_summary


def save_results(df_verbose: DataFrame, df_summary: DataFrame, ins_outs: list[dict[str, str]], out_path: str):
    logger.info(f"Trying to save results to {out_path}")
    try:
        os.makedirs(out_path, exist_ok=True)

        csv_path = os.path.join(out_path, "energy.csv")
        df_verbose.to_csv(csv_path, sep=";", index=False, mode="w", header=True)
        logger.info(f"Successfully saved verbose csv file to {csv_path}")

        csv_path = os.path.join(out_path, "..", "summary.csv")
        if not os.path.exists(csv_path):
            df_summary.to_csv(csv_path, sep=";", index=False, mode="w", header=True)
        else:
            df_summary.to_csv(csv_path, sep=";", index=False, mode="a", header=False)

        logger.info(f"Successfully saved summary csv file to {csv_path}")

        json_path = os.path.join(out_path, "ins_outs.json")
        with open(json_path, "w", encoding="utf8") as file:
            json.dump(ins_outs, file, indent=4)
        logger.info(f"Successfully saved json file to {json_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
