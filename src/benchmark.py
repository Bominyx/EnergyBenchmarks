import argparse


def main():
    parser = argparse.ArgumentParser(description="Energy Benchmark - Highly recommended to read the README first!")

    parser.add_argument("-m", "--model", required=True, type=str,
                        help="HuggingFace repository ID of a model")

    parser.add_argument("-b", "--batch-size", required=True, type=int, nargs="+",
                        help="One or multiple integers specifying the batch size")

    parser.add_argument("-t", "--max-new-token", required=True, type=int,
                        help="The number of new tokens to generate")

    parser.add_argument("-d", "--device", required=True, type=str,
                        help="The device on which the model should run")

    parser.add_argument("-p", "--path", required=False, type=str,
                        help="Path were to save the results. When none, safes results in the current working directory")

    parser.add_argument("-w", "--warmup", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether the model should do warmup generation or not. By default, true")

    parser.add_argument("-s", "--seed", default=42, type=int,
                        help="Seed to change what output is generated")

    dataset_group = parser.add_argument_group("Dataset options",
                                              description="Options to manipulate what dataset is used and how its processed for the benchmark")

    # no shorthand to make it unambiguous
    dataset_group.add_argument("--dataset-path", required=False, type=str,
                               help="Path were the benchmark dataset is located on disk. Takes priority over other dataset options")

    dataset_group.add_argument("--hf-repo-id", default="agentlans/high-quality-english-sentences", type=str,
                               help="HuggingFace repository ID of a dataset")

    dataset_group.add_argument("--dataset-split", default="test", type=str,
                               help="Dataset split name")

    dataset_group.add_argument("--measure-samples", default=250, type=int,
                               help="How many samples to draw from each bucket")

    dataset_group.add_argument("--warmup-samples", default=20, type=int,
                               help="How many warmup samples to draw")

    args = parser.parse_args()

    # local import so it only gets loaded when actually used and the help call doesn't take too long to display
    import energy
    energy.bench(args)


if __name__ == "__main__":
    main()
