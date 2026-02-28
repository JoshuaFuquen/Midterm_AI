import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class Config:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 128
    val_split: float = 0.1
    results_dir: Path = Path("results")
    metrics_file: Path = Path("results/metrics.txt")


def set_seed(seed: int) -> None:
    """Make runs more reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add channel dim for CNN: (28, 28) -> (28, 28, 1)
    x_train = (x_train.astype(np.float32) / 255.0)[..., None]
    x_test = (x_test.astype(np.float32) / 255.0)[..., None]

    # Basic sanity checks to avoid silent shape bugs
    assert x_train.ndim == 4 and x_train.shape[1:] == (28, 28, 1)
    assert x_test.ndim == 4 and x_test.shape[1:] == (28, 28, 1)
    assert y_train.ndim == 1 and y_test.ndim == 1

    return (x_train, y_train), (x_test, y_test)


def build_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_metrics(cfg: Config, test_loss: float, test_acc: float, train_acc: float, val_acc: float) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    text = (
        f"Test accuracy: {test_acc:.4f}\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Last epoch train_acc: {train_acc:.4f}\n"
        f"Last epoch val_acc: {val_acc:.4f}\n"
    )
    cfg.metrics_file.write_text(text, encoding="utf-8")


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_cnn()

    history = model.fit(
        x_train,
        y_train,
        validation_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])

    print("\n=== Final Evaluation ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    save_metrics(cfg, test_loss=test_loss, test_acc=test_acc, train_acc=train_acc, val_acc=val_acc)
    print(f"Saved metrics to: {cfg.metrics_file}")


if __name__ == "__main__":
    main()