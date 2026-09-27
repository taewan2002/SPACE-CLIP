# Reproducing the controlled v1 study

Run commands from the repository root. The CPU archive-verification path in the [README](../README.md) is independent of training. This guide describes reproducing predictions from the underlying data.

## Environment and data

The experiments used Linux, Python 3.12.3, PyTorch 2.10.0, Transformers 5.2.0, and an NVIDIA RTX 5090. The complete study dependency pins are in `requirements-study.txt`. Install a PyTorch wheel appropriate for your CUDA driver, then install the pinned requirements. No mixed precision or TF32 is used in this study. Exact training trajectories can vary with hardware/software despite fixed seeds.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-study.txt
export HF_HOME="$PWD/.hf_cache"
python -m scripts.layer_study.prepare_backbone
export HF_HUB_OFFLINE=1
```

The helper downloads `openai/clip-vit-base-patch16` at revision `57c216476eefef5ab752ec549e440a49ae4ae5f3` and pins the local default reference. Use a dedicated cache; it changes `refs/main` inside that cache. Training saves and checks the frozen-backbone state hash. Raw Fourier analysis also verifies the backbone against the archived study hash.

Obtain NYU Depth V2 RGB/depth files from the dataset provider using the [dataset layout guide](../datasets/README.md). The controlled split uses the repository's existing 24,231-frame training list and 654-image official test list. RGB and depth paths are relative to the two roots below; depth PNGs are interpreted in millimetres by the NYU loader.

```bash
python -m scripts.layer_study.prepare \
  --train-root /path/to/nyu_depth_v2/sync \
  --test-root /path/to/nyu_depth_v2/official_splits/test \
  --audit
```

Preparation hashes the original split lists, groups training captures by scene directory with a trailing letter removed, and uses seed `20260926` to hold out 10% of groups. The resulting train/validation counts are 21,974/2,257 images across 224/25 groups. Test paths are category-based and are not treated as physical scene identities. The official test list is preserved byte for byte.

Templates in `configs/layer_study/` remain unchanged. Machine-local absolute paths and generated splits go under ignored `study/`. Setup is idempotent; changing an existing setup after a run starts is rejected. Use a fresh checkout for a different protocol. Avoid running preparation concurrently with experiments.

## Smoke tests, training, and official test

Select a GPU using `CUDA_VISIBLE_DEVICES`. The archived configuration uses at most 40% of that device's memory, four CPU threads, four training workers, and two evaluation workers. Adjusting batch size or other learning/evaluation settings defines a different experiment; use a separate checkout and report the changes.

```bash
export CUDA_VISIBLE_DEVICES=0
for variant in early middle late; do
  python -m scripts.layer_study.run --variant "$variant" --smoke
done
python -m scripts.layer_study.run_queue
```

A smoke test performs one real-data training batch and evaluates four validation images. Smoke products are isolated under `study/smoke/`. The queue checks matching initial decoder/backbone hashes and parameter counts, trains all three arms, and only then evaluates each selected checkpoint on the official test set. It runs in the foreground and does not create a scheduler, daemon, or notification timer. It uses a POSIX file lock, so Linux is the supported training platform.

All arms use seed 42, 20 epochs, batch size 32, float32, AdamW (learning rate `1e-4`, weight decay `0.01`), 5% linear warmup followed by cosine decay to `1e-6`, gradient clipping at 1, and decoder EMA decay `0.996`. The main loss is equal-weight SILog and SSIM with auxiliary SILog weights 0.10 and 0.05. The frozen backbone contains 85,799,424 parameters; the decoder contains 12,050,996 trainable parameters.

The queue can be invoked again after interruption. Completed phases are skipped and training resumes from the last saved epoch. Configuration hashes must agree. Checkpoints use PyTorch serialization with optimizer state; only load checkpoints you trust. Dataset/cache paths are part of the hash, so copying an existing checkpoint into a different local configuration is not a supported resume procedure.

Outputs include `study/runs/<variant>/best.pt`, `last.pt`, `history.jsonl`, `initialization.json`, `training_complete.json`, `test_metrics.json`, and `test_per_image.jsonl`. The archived numeric release omits checkpoints. With completed training, a direct official-test rerun is available through `python -m scripts.layer_study.run --variant early --phase test`; the queue is the preferred workflow because it enforces the all-arms-first ordering.

## Evaluation protocol

The raw or EMA checkpoint with the lowest validation AbsRel is selected. Validation uses no flip averaging. The official test uses horizontal-flip prediction averaging, native 480 × 640 ground truth, the crop `[45:471, 41:601]`, and valid depths strictly between 0.001 and 10 m. Predictions are clipped to that range without ground-truth median scaling. Metrics are averaged across images, with per-region image/pixel denominators retained.

Boundary pixels are defined by adjacent valid GT log-depth differences greater than 0.05, dilated with a 7 × 7 square (radius three). A matching erosion of the valid mask excludes neighborhoods of missing depth and crop borders. Thresholds 0.03 and 0.10 are sensitivity checks. RGB edges do not define the masks.

This is a single-seed, within-protocol comparison. Test folders do not identify physical scenes, so no test-scene bootstrap is claimed. Historical release numbers use different checkpoint-selection and scaling conditions; see [legacy_results.md](legacy_results.md).

## Fourier analysis

After training, run:

```bash
python -m scripts.layer_study.fourier_analysis --phase raw --device cpu
for variant in early middle late; do
  python -m scripts.layer_study.fourier_analysis --phase trained --variant "$variant" --device cuda
done
python -m scripts.layer_study.build_results --study study --output study/reproduced-figures
```

The default archive input of `build_results` is `results/nyu_layer_study`; pass `--study study` explicitly to plot your new inference results. For short inference smoke checks, use `--phase raw --max-images 2`, or `--phase trained --variant early --smoke-checkpoint --max-images 2`. Partial outputs use separate smoke directories.

Selection deterministically takes eight hash-ordered images from each of the 25 validation scenes. Raw spectra cover all 13 CLIP hidden states; projected spectra cover seven decoder features before spatial upsampling. Each channel has its window-weighted mean removed, powers are summed across channels, and AC energy is normalized independently per image/feature. The primary window is Hann; rectangular-window results and absolute-energy uniform-input controls are retained. Frequencies are cycles per input field: low `0 < r < 2`, middle `2 ≤ r < 4`, high `r ≥ 4`, including diagonal frequencies through `sqrt(98)`.

Perturbations replace only the three additional-pathway patch maps after the frozen encoder. CLS tokens, channel means, main-pathway maps, and global conditioning remain fixed. Conditions are baseline, FFT round trip, low-pass, energy-matched low-pass, and energy-matched high-pass. Transforms use float64 and return float32 features. Filtering can induce out-of-distribution features and ringing; it is a sensitivity analysis.

Scene means receive equal weight. Intervals use 2,000 scene-bootstrap samples with seed `20260926`; the post-hoc boundary-minus-interior contrast uses seed `20260927`, reports all nine contrasts, and has no multiplicity adjustment. Validation scenes also supported checkpoint selection, so these are exploratory analyses. No repeated-seed confidence interval is implied.

## Qualitative figure

Five official-test examples were fixed by a hash policy before final test results; the policy is archived in `results/nyu_layer_study/qualitative_sample_policy.json`. After training and test evaluation:

```bash
python -m scripts.layer_study.export_qualitative
python -m scripts.layer_study.build_qualitative
```

The exporter verifies each prediction's AbsRel against the full-test record before saving local RGB/GT/prediction arrays. Dataset images and trained weights are not added to the public numeric archive.
