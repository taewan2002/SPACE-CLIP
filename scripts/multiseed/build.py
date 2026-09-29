"""Recompute five-seed statistics, LaTeX tables, and figures without a GPU."""

import argparse
import json
from pathlib import Path

from scripts.multiseed import aggregate, spectra
from scripts.multiseed.figures import perturbation_figure, repeatability_figure, spectra_figure
from scripts.multiseed.tables import write_tables


def build(data=aggregate.DATA, output=aggregate.OUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output == data or data in output.parents:
        raise ValueError("Generated outputs must not overwrite the versioned numeric archive")
    output.mkdir(parents=True, exist_ok=True)
    aggregate.generate(data, output)
    spectra.generate(data, output)
    numbers = json.loads((output / "manuscript_numbers.json").read_text())
    pooled = json.loads((output / "projected_pooled.json").read_text())
    frozen = json.loads((data / "frozen_feature_summary.json").read_text())
    selection = {
        variant: {
            seed: json.loads((data / str(seed) / variant / "test_metrics.json").read_text())
            for seed in numbers["seeds"]
        }
        for variant in aggregate.VARIANTS
    }
    write_tables(numbers, selection, output)
    perturbation_figure(numbers, output)
    repeatability_figure(numbers, output)
    spectra_figure(frozen, pooled, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=aggregate.DATA)
    parser.add_argument("--output", type=Path, default=aggregate.OUT)
    args = parser.parse_args()
    print("Generated:", build(args.data, args.output))


if __name__ == "__main__":
    main()
