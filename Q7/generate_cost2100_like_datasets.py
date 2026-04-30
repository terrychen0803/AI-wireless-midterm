"""
Exercise 2.15 dataset generator.

This script creates more than five COST2100-like channel datasets by changing
UE position distributions. It outputs MATLAB .mat files whose variable name is
`HT`, compatible with the CsiNet-style training script in this folder.

Important:
- This is a pure-Python geometry/cluster-based approximation for homework
  experimentation when the official COST2100 MATLAB generator is not available.
- For strict COST2100 reproduction, generate H using the official COST2100
  package and save the same `HT` arrays in each dataset folder.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.io import savemat

IMG_H = 32
IMG_W = 32
IMG_C = 2


@dataclass(frozen=True)
class Scenario:
    name: str
    sampler: str
    area_side_m: float = 20.0
    n_clusters: int = 4
    delay_spread: float = 1.8
    angle_spread: float = 2.0
    shadow_std_db: float = 3.0
    path_loss_exp: float = 2.0


def sample_users(rng: np.random.Generator, n: int, scenario: Scenario) -> np.ndarray:
    """Return UE positions [n,2] in meters, with BS at (0,0)."""
    side = scenario.area_side_m
    half = side / 2
    name = scenario.sampler

    if name == "cell_uniform":
        xy = rng.uniform(-half, half, size=(n, 2))

    elif name == "center_uniform":
        r = half * 0.30 * np.sqrt(rng.random(n))
        th = rng.uniform(-np.pi, np.pi, size=n)
        xy = np.column_stack([r * np.cos(th), r * np.sin(th)])

    elif name == "edge_uniform":
        # Rejection sampling for outer ring of the square cell.
        pts = []
        while len(pts) < n:
            cand = rng.uniform(-half, half, size=(max(1024, n), 2))
            r = np.linalg.norm(cand, axis=1)
            keep = cand[r > 0.65 * half]
            pts.extend(keep.tolist())
        xy = np.asarray(pts[:n], dtype=np.float32)

    elif name == "left_half":
        xy = np.column_stack([
            rng.uniform(-half, 0.0, size=n),
            rng.uniform(-half, half, size=n),
        ])

    elif name == "right_half":
        xy = np.column_stack([
            rng.uniform(0.0, half, size=n),
            rng.uniform(-half, half, size=n),
        ])

    elif name == "two_hotspots":
        centers = np.array([[-0.55 * half, -0.30 * half], [0.55 * half, 0.35 * half]])
        idx = rng.integers(0, 2, size=n)
        xy = centers[idx] + rng.normal(scale=0.13 * side, size=(n, 2))
        xy = np.clip(xy, -half, half)

    elif name == "diagonal_corridor":
        t = rng.uniform(-half, half, size=n)
        xy = np.column_stack([t, t]) + rng.normal(scale=0.06 * side, size=(n, 2))
        xy = np.clip(xy, -half, half)

    else:
        raise ValueError(f"Unknown sampler: {name}")

    return xy.astype(np.float32)


def synthesize_channel_from_positions(
    rng: np.random.Generator,
    xy: np.ndarray,
    scenario: Scenario,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> np.ndarray:
    """Create sparse angular-delay CSI matrices H in complex baseband."""
    n = xy.shape[0]
    d_grid = np.arange(img_h)[:, None]
    a_grid = np.arange(img_w)[None, :]
    H = np.zeros((n, img_h, img_w), dtype=np.complex64)

    dist = np.linalg.norm(xy, axis=1) + 1.0
    user_angle = np.arctan2(xy[:, 1], xy[:, 0])
    base_angle_bin = (np.sin(user_angle) + 1.0) * 0.5 * (img_w - 1)
    base_delay_bin = np.clip((dist / (scenario.area_side_m / np.sqrt(2))) * 18.0 + 1.0, 0, img_h - 1)

    path_loss = dist ** (-scenario.path_loss_exp)
    shadow = 10 ** (rng.normal(0.0, scenario.shadow_std_db, size=n) / 20.0)
    large_scale = path_loss * shadow

    for k in range(scenario.n_clusters):
        # Local clusters around the UE-dependent dominant delay/angle.
        delay_center = base_delay_bin + rng.normal(0.0, 2.0 + 0.3 * k, size=n)
        angle_center = base_angle_bin + rng.normal(0.0, 3.0 + 0.5 * k, size=n)
        delay_center = np.clip(delay_center, 0, img_h - 1)
        angle_center = np.mod(angle_center, img_w)

        amp = large_scale * rng.rayleigh(scale=1.0, size=n) * np.exp(-0.55 * k)
        phase = rng.uniform(-np.pi, np.pi, size=n)
        coeff = amp * np.exp(1j * phase)

        for i in range(n):
            # Circular distance for angular bins.
            da = np.minimum(np.abs(a_grid - angle_center[i]), img_w - np.abs(a_grid - angle_center[i]))
            dd = d_grid - delay_center[i]
            blob = np.exp(-0.5 * ((dd / scenario.delay_spread) ** 2 + (da / scenario.angle_spread) ** 2))
            H[i] += coeff[i] * blob.astype(np.complex64)

    # Add a weak diffuse component.
    diffuse = (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)).astype(np.complex64)
    H += 0.01 * np.mean(np.abs(H), axis=(1, 2), keepdims=True) * diffuse

    # Normalize each sample, then shift real/imag to [0,1] as in CsiNet datasets.
    max_abs = np.max(np.abs(H), axis=(1, 2), keepdims=True) + 1e-12
    H = 0.45 * H / max_abs
    return H.astype(np.complex64)


def complex_to_ht(H: np.ndarray) -> np.ndarray:
    """Convert complex H [N,32,32] to CsiNet HT [N, 2048] in [0,1]."""
    stacked = np.stack([H.real + 0.5, H.imag + 0.5], axis=-1).astype(np.float32)
    return stacked.reshape((H.shape[0], -1))


def make_dataset(
    out_dir: Path,
    scenario: Scenario,
    n_train: 1000,
    n_val: 200,
    n_test: 200,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    ds_dir = out_dir / scenario.name
    ds_dir.mkdir(parents=True, exist_ok=True)

    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        xy = sample_users(rng, n, scenario)
        H = synthesize_channel_from_positions(rng, xy, scenario)
        HT = complex_to_ht(H)
        savemat(ds_dir / f"{split}.mat", {"HT": HT, "H_complex": H, "UE_xy": xy})
        print(f"saved {ds_dir / f'{split}.mat'}  HT={HT.shape}")


def default_scenarios() -> Dict[str, Scenario]:
    return {
        "cell_uniform": Scenario("cell_uniform", "cell_uniform"),
        "center_uniform": Scenario("center_uniform", "center_uniform", n_clusters=3, delay_spread=1.5),
        "edge_uniform": Scenario("edge_uniform", "edge_uniform", n_clusters=5, delay_spread=2.2),
        "left_half": Scenario("left_half", "left_half"),
        "right_half": Scenario("right_half", "right_half"),
        "two_hotspots": Scenario("two_hotspots", "two_hotspots", n_clusters=4, angle_spread=1.6),
        "diagonal_corridor": Scenario("diagonal_corridor", "diagonal_corridor", n_clusters=4, angle_spread=1.4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--datasets", nargs="*", default=["all"], help="dataset names or all")
    args = parser.parse_args()

    scenarios = default_scenarios()
    names = list(scenarios) if "all" in args.datasets else args.datasets
    for idx, name in enumerate(names):
        if name not in scenarios:
            raise ValueError(f"Unknown dataset {name}. Valid: {list(scenarios)}")
        make_dataset(args.out, scenarios[name], args.n_train, args.n_val, args.n_test, args.seed + 97 * idx)


if __name__ == "__main__":
    main()
