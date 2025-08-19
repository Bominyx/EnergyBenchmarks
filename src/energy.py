import random
from time import time
from os import makedirs, getcwd
from os.path import join, exists

import pandas as pd
import torch
import json

from datasets import load_dataset, DatasetDict, load_from_disk
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from zeus.monitor import ZeusMonitor
from zeus.utils.logging import get_logger

# re-using logger from zeus to keep console log uniform
logger = get_logger(name=__name__)


def bench(args):
    """Driver code for the energy benchmark

    :param args: arguments parsed by argparse
    :return:
    """
    logger.info(f"Running energy benchmark with {args.model}")

    monitor = ZeusMonitor(
        gpu_indices=args.gpu_indices,
        cpu_indices=args.cpu_indices,
        approx_instant_energy=args.approx_instant_energy)

    tokenizer = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)

    if args.dataset_path:
        logger.info("Loading benchmark dataset from disk")
        dataset = load_from_disk(args.dataset_path)
    else:
        logger.info("Creating benchmark dataset in-memory")
        dataset = create_warmup_measure_dataset(tokenizer, args.hf_repo_id, args.dataset_split,
                                                args.measure_samples, args.warmup_samples)

    if args.path:
        base_out_path = join(args.path, args.model.replace("/", "_"))
    else:
        base_out_path = join(getcwd(), args.model.replace("/", "_"))

    if args.warmup:
        logger.info("Starting warmup")
        prompts = dataset["warmup"]["text"]

        # warmup is only done once. first batch size suffices
        for i in range(0, len(prompts), args.batch_size[0]):
            # slice includes only the remaining elements if it exceeds the list length
            batch_prompts = prompts[i:i + args.batch_size[0]]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            model.generate(**inputs,
                           do_sample=True,
                           temperature=1,
                           max_new_tokens=args.max_new_token,
                           pad_token_id=tokenizer.eos_token_id)

        logger.info("Warmup finished")

    prompts = dataset["measurement"]["text"]

    torch.manual_seed(args.seed)
    with torch.no_grad():
        for batch_size in args.batch_size:
            logger.info(f"Benchmarking with batch size {batch_size}")
            inputs_outputs = []
            measurements = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

                # accessing timestamps of a ZeusMonitor is troublesome
                begin = time()
                monitor.begin_window("generate", sync_execution=True)
                outputs = model.generate(**inputs,
                                         do_sample=True,
                                         temperature=1,
                                         max_new_tokens=args.max_new_token,
                                         pad_token_id=tokenizer.eos_token_id)
                measurement = monitor.end_window("generate", sync_execution=True)
                end = time()

                for prompt, completion in zip(batch_prompts, outputs):
                    inputs_outputs.append({
                        "prompt": prompt,
                        "completion": tokenizer.decode(completion, skip_special_tokens=True)[len(prompt):]
                    })

                # length of input (all are same length due to padding)
                inputs_length = len(inputs[0])
                # remove the prompt part from the output to keep only the completion
                completion = outputs[0][inputs_length:]
                # total generated tokens for the batch
                tokens = len(batch_prompts) * len(completion)

                measurements.append({
                    "measurement": measurement, "start": begin, "end": end, "tokens": tokens
                })

            logger.info(f"Benchmark with batch size {batch_size} finished")
            run_folder = f"batch-size{batch_size}_token{args.max_new_token}"
            dataframe_verbose, dataframe_summary = create_dataframes(measurements, batch_size, run_folder)
            save_results(dataframe_verbose, dataframe_summary, inputs_outputs, join(str(base_out_path), run_folder))


def create_warmup_measure_dataset(tokenizer: PreTrainedTokenizerBase, hf_dataset_path: str, dataset_split: str,
                                  sample_per_bucket: int, warmup_samples: int) -> DatasetDict:
    """Create the benchmark dataset containing a warmup and measurement split.

    Tokenizes the dataset, draws samples based on token length for each of 4 buckets and sorts the data ascending by token length.
    Additional samples from the first bucket are drawn for warmup.

    :param tokenizer: Tokenizer to tokenize the text
    :param hf_dataset_path: Path to a HuggingFace repository identifier
    :param dataset_split: Name of the dataset split e.g. test
    :param sample_per_bucket: How many samples for measurements to draw from each bucket
    :param warmup_samples: How many samples for warmup to draw from the first bucket
    :return: Dataset containing a warmup and measurement split
    """
    dataset = load_dataset(hf_dataset_path, split=dataset_split)

    token_lengths = [len(tokenizer.encode(x["text"], add_special_tokens=False)) for x in dataset]

    buckets = {
        "A": {"min": 0, "max": 15, "indices": []},
        "B": {"min": 16, "max": 30, "indices": []},
        "C": {"min": 31, "max": 50, "indices": []},
        "D": {"min": 51, "max": 100, "indices": []},
    }

    for idx, length in enumerate(token_lengths):
        for bucket in buckets.values():
            if bucket["min"] <= length <= bucket["max"]:
                bucket["indices"].append(idx)
                break

    random.seed(42)
    final_indices = []
    for name, bucket in buckets.items():
        if name == "A":
            # bucket samples and warmup samples are drawn together to make sure every value is unique
            final_indices.extend(random.sample(bucket["indices"], sample_per_bucket + warmup_samples))
        else:
            final_indices.extend(random.sample(bucket["indices"], sample_per_bucket))

    warmup_indices = final_indices[:warmup_samples]
    final_indices = final_indices[warmup_samples:]

    warmup_indices = sorted(warmup_indices, key=lambda i: token_lengths[i])
    final_indices = sorted(final_indices, key=lambda i: token_lengths[i])

    warmup_split = dataset.select(warmup_indices)
    measurement_split = dataset.select(final_indices)

    dataset_dict = DatasetDict({
        "warmup": warmup_split,
        "measurement": measurement_split
    })

    return dataset_dict


def create_dataframes(measurements: list[dict], batch_size: int, run_folder: str) -> tuple[DataFrame, DataFrame]:
    """Builds a verbose dataframe and a summary dataframe from measurements.

    :param measurements: A list of dictionaries containing the zeus measurements, start and end timestamps of the measurements and the generated token count
    :param batch_size: The batch size
    :param run_folder: The folder were the verbose dataframe will be saved
    :return: The dataframe containing the exact measurements of each batch and the dataframe summarizing the measurements of all batches of that batch size
    """
    dataframe_verbose = pd.DataFrame(columns=["time(s)", "cpu_energy(J)", "gpu_energy(J)", "start", "end", "tokens"])

    for _measurement in measurements:
        measurement = _measurement["measurement"]
        start = _measurement["start"]
        end = _measurement["end"]
        tokens = _measurement["tokens"]
        if not measurement.cpu_energy:
            # if zeus can't find any cpu, the cpu_energy dict is none, so here it's manually set to 0
            measurement.cpu_energy = {0: 0}

        dataframe_verbose.loc[len(dataframe_verbose)] = {
            "time(s)": measurement.time,
            "cpu_energy(J)": sum(
                measurement.cpu_energy.values()),
            "gpu_energy(J)": sum(
                measurement.gpu_energy.values()),
            "start": start,
            "end": end,
            "tokens": tokens
        }

    total_time = dataframe_verbose["time(s)"].sum()
    avg_time = dataframe_verbose["time(s)"].mean()

    total_cpu_energy = dataframe_verbose["cpu_energy(J)"].sum()
    avg_cpu_energy = dataframe_verbose["cpu_energy(J)"].mean()

    total_gpu_energy = dataframe_verbose["gpu_energy(J)"].sum()
    avg_gpu_energy = dataframe_verbose["gpu_energy(J)"].mean()

    total_tokens = dataframe_verbose["tokens"].sum()
    avg_tokens = dataframe_verbose["tokens"].mean()

    dataframe_summary = pd.DataFrame({
        "run": run_folder,
        "total_time(s)": [total_time],
        "avg_time(s)": [avg_time],
        "total_cpu_energy(J)": [total_cpu_energy],
        "avg_cpu_energy(J)": [avg_cpu_energy],
        "total_gpu_energy(J)": [total_gpu_energy],
        "avg_gpu_energy(J)": [avg_gpu_energy],
        "total_tokens": [total_tokens],
        "avg_tokens": [avg_tokens],
        "batch_size": [batch_size],
    })
    return dataframe_verbose, dataframe_summary


def save_results(dataframe_verbose: DataFrame, dataframe_summary: DataFrame, ins_outs: list[dict[str, str]],
                 out_path: str):
    """Saves the dataframes and list of dictionaries with the inputs and outputs to the given path.

    :param dataframe_verbose: The dataframe containing the exact measurements of each batch
    :param dataframe_summary: The dataframe summarizing the measurements of all batches of one size
    :param ins_outs: A list of dictionaries containing the input prompts and the output completions
    :param out_path: The output path
    :return:
    """
    logger.info(f"Trying to save results to {out_path}")
    try:
        makedirs(out_path, exist_ok=True)

        csv_path = join(out_path, "energy.csv")
        dataframe_verbose.to_csv(csv_path, sep=";", index=False, mode="w", header=True)
        logger.info(f"Successfully saved verbose csv file to {csv_path}")

        csv_path = join(out_path, "..", "summary.csv")
        if not exists(csv_path):
            dataframe_summary.to_csv(csv_path, sep=";", index=False, mode="w", header=True)
        else:
            dataframe_summary.to_csv(csv_path, sep=";", index=False, mode="a", header=False)
        logger.info(f"Successfully saved summary csv file to {csv_path}")

        json_path = join(out_path, "ins_outs.json")
        with open(json_path, "w", encoding="utf8") as file:
            json.dump(ins_outs, file, indent=4)
        logger.info(f"Successfully saved json file to {json_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
