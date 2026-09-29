# Five-seed reproduction

The final controlled study comprises seeds 42–46 and three layer variants per seed. The initial seed-42 test results preceded the additional runs. All 15 checkpoints were selected using validation AbsRel; test scores were not used to select checkpoints. The less stable early seed-44 run remains in every five-seed aggregate.

## Reproduce archived statistics on CPU

```bash
python -m pip install -r requirements-analysis.txt
python -m scripts.multiseed.verify
python -m scripts.multiseed.build
```

The default output is `study/multiseed-figures/`. Use `--data` and `--output` to change the input archive or output directory. The builder rejects writing into its input archive. The raw record hashes, validation minima, shared initializations within each seed, independent initializations across seeds, official test identities, perturbation sample identities, and derived statistics are checked by the verifier.

The builder writes `manuscript_numbers.json`, `projected_pooled.json`, five LaTeX tables, and three figures (`feature_spectra`, `frequency_perturbation`, `seed_repeatability`) in PDF and PNG. Frozen-backbone spectra are shared across seeds and come from the original frozen-feature archive; its per-image verification remains available with `python -m scripts.layer_study.verify_results`.

## Train independently

Follow [reproducibility.md](reproducibility.md) in a separate checkout for each seed. Keep the data roots, frozen backbone revision, scene split, and hyperparameters fixed. For example, in the seed-43 checkout:

```bash
python -m scripts.layer_study.prepare --seed 43 \
  --train-root /path/to/nyu_depth_v2/sync \
  --test-root /path/to/nyu_depth_v2/official_splits/test --audit
CUDA_VISIBLE_DEVICES=0 python -m scripts.layer_study.run_queue
```

Repeat for seeds 42, 44, 45, and 46. The queue trains the three variants and evaluates their validation-selected checkpoints within that checkout. It resumes interrupted training and prevents changes to an existing run's setup. Do not switch the seed in a checkout that contains started runs. The queue's within-seed ordering does not describe the historical order of the entire extension.

For each seed, run trained Fourier inference as documented in the reproduction guide. To aggregate new measurements, arrange each run under `<archive>/<seed>/<variant>/`: `test_metrics.json`, `test_per_image.jsonl`, `perturbation_per_image.jsonl`, and `projected_per_image_{hann,rectangular}.jsonl`. Copy the shared frozen summary to `<archive>/frozen_feature_summary.json`. The reader supports plain or losslessly gzipped JSONL. Run `python -m scripts.multiseed.build --data <archive> --output <output>`; this generates fresh statistics without comparing them to the published values. The archive verifier is for the released records and additionally requires its training provenance and checksum inventory.

## Interpretation

Test means are computed over 654 images within each seed, then summarized with the mean and sample SD across five seeds. Perturbations are paired with the same seed's unmodified predictions. Scene-bootstrap intervals use 2,000 resamples of 25 validation scenes after averaging the paired per-image effects across seeds. They quantify scene sampling, not uncertainty over training seeds. These scenes also supported checkpoint selection. Regional contrasts are exploratory and are not multiplicity-adjusted. No new controlled KITTI training or evaluation is included in this release.
