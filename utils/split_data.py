import os
import shutil

import yaml
from sklearn.model_selection import train_test_split


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

SEED = config["SEED"]


def split_data(root: str, seed=SEED):
    images = os.listdir(root)
    train, test = train_test_split(images, test_size=0.2, shuffle=True, random_state=seed)
    test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=seed)

    train_path = os.path.join(root, "train")
    test_path = os.path.join(root, "test")
    val_path = os.path.join(root, "val")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    assert len(test) + len(train) + len(val) == len(images)
    assert len(set(train) & set(test)) == 0
    assert len(set(train) & set(val)) == 0
    assert len(set(test) & set(val)) == 0

    for item in train:
        current_path = os.path.join(root, item)
        dest_path = os.path.join(train_path, item)
        print(f"Moving {current_path} to {dest_path}")
        shutil.move(current_path, dest_path)
    for item in test:
        current_path = os.path.join(root, item)
        dest_path = os.path.join(test_path, item)
        print(f"Moving {current_path} to {dest_path}")
        shutil.move(current_path, dest_path)
    for item in val:
        current_path = os.path.join(root, item)
        dest_path = os.path.join(val_path, item)
        print(f"Moving {current_path} to {dest_path}")
        shutil.move(current_path, dest_path)


def check_splitting(root: str):
    train = os.listdir(os.path.join(root, "train"))
    test = os.listdir(os.path.join(root, "test"))
    val = os.listdir(os.path.join(root, "val"))

    assert len(set(train) & set(test)) == 0
    assert len(set(train) & set(val)) == 0
    assert len(set(test) & set(val)) == 0

    print("Data split is correct")


if __name__ == "__main__":
    # split_data("../data")
    check_splitting("../data")
