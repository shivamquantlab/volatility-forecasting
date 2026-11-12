"""Create a fixed-seed synthetic volatility regime and EWMA chart."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ewma import ewma_variance


def run(output_dir: Path, seed: int = 17) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns = np.concatenate((
        rng.normal(0.0, 0.006, 126),
        rng.normal(0.0, 0.018, 126),
    ))
    states = ewma_variance(returns, decay=0.94)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.sqrt(states[:-1] * 252), color="#7c3aed", lw=1.4,
            label="EWMA volatility known before return")
    ax.axvline(126, color="#64748b", ls="--", lw=1, label="Synthetic regime boundary")
    ax.set(title="One-step EWMA volatility forecast", xlabel="Trading-day index",
           ylabel="Annualized volatility")
    ax.legend(frameon=False)
    ax.text(0.01, 0.98, "SYNTHETIC • seed 17 • not investment performance",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ewma_volatility.png", dpi=150)
    plt.close(fig)
    summary = {
        "seed": seed,
        "decay": 0.94,
        "next_step_annualized_volatility": float(np.sqrt(states[-1] * 252)),
        "observations": int(returns.size),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic EWMA report to {args.output_dir}")


if __name__ == "__main__":
    main()

