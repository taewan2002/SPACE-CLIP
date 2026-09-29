# SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation

[Paper (arXiv)](https://arxiv.org/abs/2601.17657) · [Reproduction guide](docs/reproducibility.md) · [Five-seed measurements](results/nyu_multiseed) · [Historical release](docs/legacy_results.md)

SPACE-CLIP estimates monocular depth with a **frozen CLIP vision encoder** and a trainable dual-pathway decoder. Inference uses RGB images and no text encoder. This repository contains the original model and the controlled NYU layer-selection and frequency-analysis study for the revised **v1** manuscript. It does not implement the separate v2 study.

The main pathway uses four CLIP hidden states with global-context FiLM conditioning. An additional pathway supplies three other hidden states, and the decoder fuses both pathways to predict dense depth. The original code calls these pathways `semantic` and `structural`; these names are architectural labels, not evidence that the features encode exclusively semantics or geometry.

![SPACE-CLIP architecture](figures/fig2.png)

## Controlled NYU results

All three arms use the same architecture size, initialization, training recipe, and held-out validation scenes. Only the additional-pathway layer indices change; the main indices remain `[12, 9, 6, 3]`. `L0` denotes the embedding output and `L1`–`L12` the transformer block outputs.

| Additional layers | AbsRel ↓ | RMSE (m) ↓ | δ₁ ↑ | Boundary AbsRel ↓ |
| --- | ---: | ---: | ---: | ---: |
| Early `[2, 1, 0]` | 0.1283 ± 0.0113 | 0.4544 ± 0.0462 | 0.8325 ± 0.0389 | 0.1447 ± 0.0103 |
| Middle `[7, 5, 4]` | 0.1208 ± 0.0017 | 0.4355 ± 0.0080 | 0.8533 ± 0.0073 | 0.1393 ± 0.0015 |
| Late `[11, 10, 8]` | 0.1230 ± 0.0020 | 0.4418 ± 0.0054 | 0.8487 ± 0.0057 | 0.1432 ± 0.0023 |

Values are mean ± sample SD over **five training seeds (42–46), 15 runs in total**. Protocol: 21,974 training images, 2,257 validation images from separate scene groups, and all 654 official test images. Every checkpoint is selected by validation AbsRel, comparing raw and EMA weights. Test evaluation uses native-resolution ground truth, the NYU Eigen crop, 0.001–10 m depths, horizontal-flip averaging, and **no median scaling**. Boundary metrics use GT log-depth jumps above 0.05 with a three-pixel dilation radius. Each arm has 12,050,996 trainable parameters.

Seed 42 was trained and tested first; seeds 43–46 were added subsequently. These are a repeated-seed extension of the initial study, not a prospectively fixed 15-run experiment. All five seeds are retained, including the less stable early-layer run at seed 44. Middle layers have the lowest mean whole-image and primary-boundary error. Against late layers, middle layers improve whole-image AbsRel in three of five seeds and primary-boundary AbsRel in all five; this does not establish universal superiority. [Historical KITTI/NYU scores](docs/legacy_results.md) use different selection/evaluation conditions and are not directly comparable with this table.

![Five-seed repeatability](figures/layer-study/seed_repeatability.png)

## Layer and frequency analysis

The study measures channel-summed Fourier energy on native 14 × 14 CLIP patch grids and perturbs only the additional pathway at inference time. Analyses use 200 validation images from 25 scenes, with equal scene weighting and scene-bootstrap intervals on seed-averaged paired effects. These intervals describe scene sampling; sample SD separately describes variation across training seeds. Controls include a rectangular window, uniform input, an FFT round trip, and AC-energy-matched filters.

![Frozen and projected feature spectra](figures/layer-study/feature_spectra.png)

![Frequency perturbation sensitivity](figures/layer-study/frequency_perturbation.png)

Frequency content changes non-monotonically with depth. The early pathway is sensitive to removing high-frequency content near depth boundaries, including under energy matching. This supports a bounded sensitivity interpretation: it does not establish an exclusive semantic/geometric decomposition or a general causal mechanism. These analyses reuse validation scenes also used for checkpoint selection; the boundary-minus-interior contrast is post hoc and its intervals are not multiplicity-adjusted.

## Reproduce tables and figures on CPU

Python 3.12 is recommended. The commands below need no dataset, model weights, GPU, or Hugging Face download.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m scripts.multiseed.verify
python -m scripts.multiseed.build
```

Outputs are written to ignored `study/multiseed-figures/`: three PDF/PNG figures, five LaTeX tables, and full-precision JSON statistics. The verifier checks archive hashes, validation-selected checkpoints in all 15 training histories, official-test metrics, shared sample identities, initialization controls, 3,000 FFT round-trip checks, and pooled statistics against the per-image records. See the [five-seed guide](docs/multiseed.md). The original seed-42 archive remains available in `results/nyu_layer_study`.

For CUDA training, real-data smoke tests, checkpoint evaluation, and independent Fourier inference, follow the [reproduction guide](docs/reproducibility.md). Trained decoder checkpoints and dataset images are **not bundled**; regenerating predictions requires training the three arms for each of five seeds. The pinned CLIP backbone can be downloaded with the supplied helper.

## Repository layout

- `space_clip.py`, `utils/`: original model, data loader, and losses.
- `scripts/layer_study/`: controlled training, evaluation, spectra, result verification, and figure generation.
- `configs/layer_study/`: portable templates; local paths are generated under ignored `study/configs/`.
- `scripts/multiseed/`: CPU verification and aggregation of all five seeds.
- `results/nyu_multiseed/`: per-image records, training histories, and summaries for 15 runs.
- `results/nyu_layer_study/`: original seed-42 records, frozen-feature analysis, split provenance, and sample identities.
- `train.py`, `configs/nyu.yaml`, `configs/kitti.yaml`, `scripts/run_release_experiment.sh`: historical training entry points; see [legacy protocol notes](docs/legacy_results.md).

Local data, weights, caches, environments, logs, and generated study products are ignored by Git. The original model, loader, and losses are byte-identical to the experimental snapshot; packaging changes are recorded in [provenance.json](results/nyu_layer_study/provenance.json).

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q scripts/layer_study scripts/multiseed
ruff check scripts/layer_study scripts/multiseed
ruff format --check scripts/layer_study scripts/multiseed
```

The CPU wheel command targets Linux/Windows; on macOS use `python -m pip install torch==2.10.0`. Tests cover metric masks, scene separation, known Fourier modes, energy/CLS preservation, setup protection, and agreement with archived results. CI also regenerates the numerical tables and figures.

## Citation and license

```bibtex
@misc{cho2026spaceclipspatialperceptionadaptive,
  title={SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation},
  author={Taewan Cho and Taeryang Kim and Andrew Jaeyong Choi},
  year={2026},
  eprint={2601.17657},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2601.17657}
}
```

Code is released under [Apache-2.0](LICENSE). Refer to the original providers for NYU/KITTI data and CLIP weights and their terms.
