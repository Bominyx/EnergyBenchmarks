import subprocess
import time
import sys


subprocess.run([
    sys.executable,
    "benchmark.py",
    "--model", "openai-community/gpt2-xl",
    "--batch-size", "1", "2", "4", "6", "8", "12", "16", "32", "64", "128", "256",
    "--max-new-token", "128",
    "--device", "cuda:0",
    "--path", "../output/"
])

time.sleep(60)
subprocess.run([
    sys.executable,
    "benchmark.py",
    "--model", "Qwen/Qwen3-1.7B-Base",
    "--batch-size", "1", "2", "4", "6", "8", "12", "16", "32", "64", "128", "256",
    "--max-new-token", "128",
    "--device", "cuda:0",
    "--path", "../output/"
])

time.sleep(60)
subprocess.run([
    sys.executable,
    "benchmark.py",
    "--model", "Qwen/Qwen3-0.6B-Base",
    "--batch-size", "1", "2", "4", "6", "8", "12", "16", "32", "64", "128", "256",
    "--max-new-token", "128",
    "--device", "cuda:0",
    "--path", "../output/"
])

time.sleep(60)
subprocess.run([
    sys.executable,
    "benchmark.py",
    "--model", "meta-llama/Llama-3.2-1B",
    "--batch-size", "1", "2", "4", "6", "8", "12", "16", "32", "64", "128", "256",
    "--max-new-token", "128",
    "--device", "cuda:0",
    "--path", "../output/"
])