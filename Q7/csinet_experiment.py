"""
Exercise 2.15 CsiNet experiment script.

Main tasks:
(a) Load more than five different COST2100/COST2100-like channel datasets.
(b) Evaluate one trained CsiNet on each dataset.
(c) Train CsiNet on the mixed datasets and compare NMSE.

The model architecture follows the CsiNet reference idea:
Conv2D encoder -> Dense codeword -> Dense decoder -> two RefineNet residual units -> Conv2D output.
This implementation uses TensorFlow 2 / Keras and channels_last for CPU/GPU compatibility.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, LeakyReLU,
                                     Add, Flatten, Dense, Reshape)

IMG_H = 32
IMG_W = 32
IMG_C = 2
IMG_TOTAL = IMG_H * IMG_W * IMG_C
ALL_DATASETS = [
    "cell_uniform",
    "center_uniform",
    "edge_uniform",
    "left_half",
    "right_half",
    "two_hotspots",
    "diagonal_corridor",
]


def load_ht(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    mat_path = data_dir / dataset / f"{split}.mat"
    mat = loadmat(mat_path)
    if "HT" not in mat:
        raise KeyError(f"{mat_path} does not contain variable `HT`.")
    x = mat["HT"].astype("float32")
    x = x.reshape((-1, IMG_H, IMG_W, IMG_C))  # channels_last
    return x


def load_many(data_dir: Path, datasets: Iterable[str], split: str) -> np.ndarray:
    arrs = [load_ht(data_dir, ds, split) for ds in datasets]
    return np.concatenate(arrs, axis=0)


def refine_block(x: tf.Tensor) -> tf.Tensor:
    shortcut = x
    y = Conv2D(8, (3, 3), padding="same")(x)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.3)(y)
    y = Conv2D(16, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.3)(y)
    y = Conv2D(2, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Add()([shortcut, y])
    y = LeakyReLU(alpha=0.3)(y)
    return y


def build_csinet(encoded_dim: int = 128, residual_num: int = 2) -> Model:
    inp = Input(shape=(IMG_H, IMG_W, IMG_C), name="CSI")

    # Encoder: convolutional feature extraction + dense compression.
    x = Conv2D(2, (3, 3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Flatten()(x)
    code = Dense(encoded_dim, activation="linear", name="codeword")(x)

    # Decoder: dense decompression + RefineNet units + sigmoid output.
    x = Dense(IMG_TOTAL, activation="linear")(code)
    x = Reshape((IMG_H, IMG_W, IMG_C))(x)
    for _ in range(residual_num):
        x = refine_block(x)
    out = Conv2D(2, (3, 3), activation="sigmoid", padding="same", name="reconstructed_CSI")(x)

    model = Model(inp, out, name=f"CsiNet_dim{encoded_dim}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def ht_to_complex(x: np.ndarray) -> np.ndarray:
    """Convert [N,32,32,2] in [0,1] to complex H [N,32*32]."""
    h = (x[..., 0] - 0.5) + 1j * (x[..., 1] - 0.5)
    return h.reshape((x.shape[0], -1))


def nmse_db(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    h_true = ht_to_complex(x_true)
    h_hat = ht_to_complex(x_hat)
    power = np.sum(np.abs(h_true) ** 2, axis=1) + 1e-12
    mse = np.sum(np.abs(h_true - h_hat) ** 2, axis=1)
    return float(10 * np.log10(np.mean(mse / power)))


def evaluate_model(model: Model, data_dir: Path, test_datasets: Iterable[str], batch_size: int) -> List[Tuple[str, float]]:
    rows = []
    for ds in test_datasets:
        x_test = load_ht(data_dir, ds, "test")
        x_hat = model.predict(x_test, batch_size=batch_size, verbose=0)
        rows.append((ds, nmse_db(x_test, x_hat)))
        print(f"test on {ds:18s} NMSE = {rows[-1][1]:8.3f} dB")
    return rows


def save_rows(rows: List[Tuple[str, float]], out_csv: Path, column_name: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", column_name])
        writer.writerows(rows)


def plot_compare(baseline_csv: Path, mixed_csv: Path, out_png: Path) -> None:
    import pandas as pd
    b = pd.read_csv(baseline_csv)
    m = pd.read_csv(mixed_csv)
    df = b.merge(m, on="dataset")
    x = np.arange(len(df))
    w = 0.38
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - w / 2, df.iloc[:, 1], width=w, label=df.columns[1])
    plt.bar(x + w / 2, df.iloc[:, 2], width=w, label=df.columns[2])
    plt.xticks(x, df["dataset"], rotation=25, ha="right")
    plt.ylabel("NMSE (dB), lower is better")
    plt.title("CsiNet generalization across channel datasets")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"saved {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--mode", choices=["baseline", "mixed", "plot"], default="baseline")
    parser.add_argument("--train-dataset", default="cell_uniform")
    parser.add_argument("--test-datasets", nargs="*", default=ALL_DATASETS)
    parser.add_argument("--encoded-dim", type=int, default=128, help="512=1/4, 128=1/16, 64=1/32, 32=1/64")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = args.result_dir / f"baseline_train_{args.train_dataset}_dim{args.encoded_dim}.csv"
    mixed_csv = args.result_dir / f"mixed_train_all_dim{args.encoded_dim}.csv"

    if args.mode == "plot":
        plot_compare(baseline_csv, mixed_csv, args.result_dir / f"compare_dim{args.encoded_dim}.png")
        return

    if args.mode == "baseline":
        train_names = [args.train_dataset]
        out_csv = baseline_csv
        tag = f"baseline_{args.train_dataset}_dim{args.encoded_dim}"
        column = "baseline_nmse_db"
    else:
        train_names = args.test_datasets
        out_csv = mixed_csv
        tag = f"mixed_all_dim{args.encoded_dim}"
        column = "mixed_nmse_db"

    print(f"training mode={args.mode}, train datasets={train_names}")
    x_train = load_many(args.data_dir, train_names, "train")
    x_val = load_many(args.data_dir, train_names, "val")

    model = build_csinet(encoded_dim=args.encoded_dim, residual_num=2)
    model.summary()

    ckpt = args.result_dir / f"{tag}.keras"
    callbacks = [
        CSVLogger(args.result_dir / f"log_{tag}.csv"),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ]

    model.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    rows = evaluate_model(model, args.data_dir, args.test_datasets, args.batch_size)
    save_rows(rows, out_csv, column)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
