# Five-seed numerical archive

This archive contains all 15 controlled NYU runs: seeds 42–46 × early/middle/late additional-pathway layers. It includes all seeds, including the less stable early seed-44 run. Seed 42 was evaluated first and the remaining seeds were added later; every checkpoint was selected by validation AbsRel.

| File | Contents |
| --- | --- |
| `<seed>/<variant>/test_metrics.json` | Official-test metrics and selected checkpoint epoch/source |
| `<seed>/<variant>/test_per_image.jsonl.gz` | Full-precision metrics for all 654 test images and regional thresholds |
| `<seed>/<variant>/history.jsonl.gz` | All 20 epochs of raw/EMA validation metrics |
| `<seed>/<variant>/initialization.json` | Model/configuration hashes, layer identities, and parameter counts |
| `<seed>/<variant>/training_complete.json` | Completion, validation selection, and frozen-backbone checks |
| `<seed>/<variant>/frequency_summary.json` | Per-seed Fourier and intervention summaries |
| `<seed>/<variant>/perturbation_per_image.jsonl.gz` | Five-condition results for the same 200 validation images |
| `<seed>/<variant>/projected_per_image_{hann,rectangular}.jsonl.gz` | Projected-feature spectra |
| `manuscript_numbers.json` | Derived five-seed test and paired perturbation statistics |
| `projected_pooled.json` | Seed-averaged projected spectra with scene-bootstrap intervals |
| `frozen_feature_summary.json` | Shared frozen CLIP analysis from the initial study |
| `run_summary.json` | Original full-precision five-seed aggregate and provenance |
| `checksums.json` | SHA-256 inventory of all 139 numeric files |

These records were copied losslessly from the completed experiments and the verified manuscript analysis. Gzip timestamps are fixed for reproducibility. Images, model weights, machine-local paths, and credentials are not included. Dataset-relative identifiers can start with `/`; this does not indicate an author machine path. Configuration hashes retain their original identities; newly generated local configurations have different hashes because their filesystem paths differ.

The shared scene split, 200-image selection, frozen-feature per-image measurements, and original code provenance are preserved in [nyu_layer_study](../nyu_layer_study). Run `python -m scripts.multiseed.verify` and `python -m scripts.multiseed.build` from the repository root. See [the guide](../../docs/multiseed.md) for independent training and statistical definitions.
