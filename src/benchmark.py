import argparse


def main():
    # TODO: Add examples either to help or README.md or both
    # TODO: When more benchmark types are added, consider adding sub-commands
    parser = argparse.ArgumentParser(description="Energy Benchmark")

    parser.add_argument('-m', '--model', required=True, type=str,
                        help="HuggingFace model with text generation capabilities")

    parser.add_argument('-b', '--batch-size', required=True, type=int, nargs="+",
                        help="One or multiple integers specifying the batch size")

    parser.add_argument('-t', '--max-new-token', required=True, type=int,
                        help="One or multiple integers specifying how many new tokens to generate")

    parser.add_argument('-d', '--device', required=True, type=str,
                        help="On which device the model should run")

    parser.add_argument('-p', '--path', required=False, type=str,
                        help="Path were to save the results. When none, safes results relative to current working directory")

    parser.add_argument('-w', '--warmup', required=False, type=bool, default=True,
                        help="If true, do warmup inference before measurements")
    args = parser.parse_args()

    # local import so it only gets loaded when actually used and the help call doesn't take too long to display
    import bench_energy
    bench_energy.bench(args)


if __name__ == "__main__":
    main()
