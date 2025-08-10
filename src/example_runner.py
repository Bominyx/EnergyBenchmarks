import subprocess
import time
import sys


# this is an example for how the benchmarks could be run automatically
# you can use this to check out if everything works correctly.
subprocess.run([
    sys.executable, "benchmark.py",
    "--model", "openai-community/gpt2-xl",
    "--batch-size", "1",
    "--max-new-token", "32", # note that the amount of new tokens to generate is set fairly low
    "--device", "cuda:0",
    "--path", "../example_output/",
    "--measure-samples", "10",
    "--warmup-samples", "5",
])

# cooldown period between benchmarks
time.sleep(5)
subprocess.run([
    sys.executable, "benchmark.py",
    "--model", "Qwen/Qwen3-0.6B-Base",
    "--batch-size", "1", "2", # multiple batch sizes can be given, so multiple benchmarks with different batch sizes are run
    "--max-new-token", "32",
    "--device", "cuda:0",
    "--path", "../example_output/",
    "--measure-samples", "10",
    "--warmup-samples", "5",
])
