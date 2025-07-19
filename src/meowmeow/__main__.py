from .model import Model


def train(args):
    model = Model(load=args.load, batch_size=args.batch_size)

    model.train(
        epochs=args.epochs,
    )

    model.save()
    print("Training complete.")


def infer(args):
    model = Model(load=True, batch_size=args.batch_size)

    predictions = model.predict(args.images)

    for image, prediction in zip(args.images, predictions):
        print(f"Image: {image}, Prediction: {prediction}")


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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference on the model")
    infer_parser.add_argument("images", nargs="+", help="Paths to images for inference")

    parse_args = parser.parse_args()

    if parse_args.command == "train":
        train(parse_args)
    elif parse_args.command == "infer":
        infer(parse_args)
    else:
        parser.print_help()
