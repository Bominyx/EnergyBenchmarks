import time
import os

import pandas as pd
import torch
import yaml
import json

from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from zeus.monitor import ZeusMonitor


# TODO: Add function comments
def bench(args):
    torch.manual_seed(42)

    file_path = hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename="Evaluation prompts.yaml",
        repo_type="dataset"
    )

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=args.device)

    with open(file_path, "r", encoding="utf8") as file:
        prompts = yaml.safe_load(file)

    monitor = ZeusMonitor()

    base_out_path = os.path.join(os.getcwd(), model_name.replace('/', '_'))

    with torch.no_grad():
        for batch_size in args.batch_size:
            for max_length in args.max_length:
                ins_outs = []
                measurements = []
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

                    # accessing zeus timestamps is troublesome
                    begin = time.time()
                    monitor.begin_window("generate")
                    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=1,
                                             pad_token_id=tokenizer.eos_token_id)
                    measurement = monitor.end_window("generate")
                    end = time.time()

                    for prompt, completion in zip(batch_prompts, outputs):
                        ins_outs.append({"prompt": prompt,
                                         "completion": tokenizer.
                                        decode(completion, skip_special_tokens=True)[len(prompt):]})

                    measurements.append({"measurement": measurement, "start": begin, "end": end,
                                         "generated_tokens": len(outputs[0][len(inputs[0]):]) * batch_size})

                df = create_dataframe(measurements)
                save_results(df, ins_outs,
                             os.path.join(base_out_path, f"batch-size{batch_size}_max-length{max_length}"))


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
