import os
import pandas


DATASETS_ROOT = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-3]), "datasets"
)

# dataset that we want to train on
TARGET_SET_PATH = os.path.join(DATASETS_ROOT, "target")

# "polution" dataset that we insert into the target set
SALT_SET_PATH = os.path.join(DATASETS_ROOT, "salt")


def load_dataset():
    if not os.path.exists(TARGET_SET_PATH):
        raise FileNotFoundError(f"Target dataset not found at {TARGET_SET_PATH}")
    if not os.path.exists(SALT_SET_PATH):
        raise FileNotFoundError(f"Salt dataset not found at {SALT_SET_PATH}")

    dataset = {
        "image": [],
        "isSilly": [],
    }

    for filename in os.listdir(TARGET_SET_PATH):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            dataset["image"].append(os.path.join(TARGET_SET_PATH, filename))
            dataset["isSilly"].append(True)

    for filename in os.listdir(SALT_SET_PATH):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            dataset["image"].append(os.path.join(SALT_SET_PATH, filename))
            dataset["isSilly"].append(False)

    return pandas.DataFrame(dataset)
