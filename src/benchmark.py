import argparse


def main():
    # TODO: Add examples either to help or README.md or both
    # TODO: Add argument so the path to a desired save location can be set
    # TODO: When more benchmark types are added, consider adding sub-commands
    parser = argparse.ArgumentParser(description="Energy Benchmark")

    parser.add_argument('-m', '--model', required=True, type=str,
                        help="HuggingFace model with text generation capabilities")

    parser.add_argument('-b', '--batch-size', required=True, type=int, nargs="+",
                        help="One or multiple integers specifying the batch size")

    parser.add_argument('-l', '--max-length', required=True, type=int, nargs="+",
                        help="One or multiple integers specifying how many new tokens to generate")

    parser.add_argument('-d', '--device', required=True, type=str,
                        help="On which device the model should run")

    args = parser.parse_args()

    # local import so it only gets loaded when actually used and the help call doesn't take too long to display.
    import ts_eval_energy
    ts_eval_energy.bench(args)


if __name__ == "__main__":
    main()
