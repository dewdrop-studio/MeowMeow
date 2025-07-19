from .model import Model


def train(args):
    model = Model(load=args.load, batch_size=args.batch_size)

    model.train(
        epochs=args.epochs,
    )

    model.save()
    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MeowMeow Model Training", add_help=True
    )

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--load",
        action="store_true",
        help="Load a pretrained model instead of training from scratch",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    parse_args = parser.parse_args()

    if parse_args.command == "train":
        train(parse_args)
    else:
        parser.print_help()
