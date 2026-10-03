# Original seed-42 measurements for the controlled v1 study

The final manuscript uses [all five seeds](../nyu_multiseed). This archive preserves the initial seed-42 analysis and the shared split, frozen-feature, and sample records.

This archive contains numerical measurements, split/sample identities, initialization records, training histories, and full-precision summaries. It contains no trained weights or dataset pixels. JSONL records are losslessly gzip-compressed with a fixed gzip timestamp.

| Path | Contents |
| --- | --- |
| `runs/<variant>/test_per_image.jsonl.gz` | 654 image-level metric records, including boundary/interior thresholds 0.03, 0.05, and 0.10 |
| `runs/<variant>/test_metrics.json` | Macro-averaged test metrics and selected checkpoint epoch/source |
| `runs/<variant>/history.jsonl.gz` | 20 epochs of raw/EMA validation metrics |
| `runs/<variant>/initialization.json` | Parameter counts, initial decoder hash, frozen-backbone hash, and original configuration hash |
| `runs/<variant>/training_complete.json` | Completion and frozen-backbone checks |
| `fourier/raw/` | Per-image raw spectra for 13 layers under Hann/rectangular windows and summary controls |
| `fourier/<variant>/` | Projected spectra and five-condition perturbation metrics for 200 validation images |
| `split_manifest.json` | Source/split hashes and scene-group assignments |
| `sample_manifest.json` | Fixed 200-image Fourier selection (25 scenes × eight images) |
| `qualitative_sample_policy.json` | Five fixed official-test examples |
| `provenance.json` | Original environment, source hashes, and packaging changes |
| `checksums.json` | SHA-256 inventory of all JSON/JSONL data files |

`image_path` identifies a path relative to the NYU dataset root, even when it starts with `/`. These are dataset identifiers, not author machine paths. Original configuration hashes include machine-local paths and therefore differ from fresh runs; the parameter and frozen-backbone hashes are the portable identity checks.

From the repository root:

```bash
python -m scripts.layer_study.verify_results
python -m scripts.layer_study.build_results
```

Verification recalculates the official-test means, validation-selected checkpoint, and raw/projected/perturbation scene-bootstrap summaries. The table/figure builder writes `study/paper-figures/`, including full-precision CSV records. This reproduces aggregation of archived measurements; independent model prediction reproduction additionally requires the data and training described in the [guide](../../docs/reproducibility.md).

Interpretation is limited to one training seed. Fourier analyses use validation scenes also used for checkpoint selection. The post-hoc regional contrasts are explicitly exploratory and unadjusted for multiplicity. The final five-seed results and their limitations are in the [README](../../README.md).
