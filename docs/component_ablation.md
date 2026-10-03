# Semantic-only control

This control asks whether adding the Structural pathway improves depth estimation over the Semantic pathway with FiLM under the controlled NYU recipe. It was specified after the 15 layer-selection runs had been evaluated. All five control trainings and official-test evaluations are complete. Numerical records and initialization audits are available in [results/nyu_component_ablation](../results/nyu_component_ablation).

## Comparison

- Seeds: 42–46, with every run retained. The primary comparison is the original early-layer dual-pathway model against its same-seed Semantic-only control. Middle and late variants are secondary comparisons.
- Retained: frozen CLIP, Semantic layers `[12, 9, 6, 3]`, FiLM, scene split, 20 epochs, optimizer, learning-rate schedule, losses, batch size, EMA, and evaluation protocol.
- Changed: `use_structural_pathway=False`. The control has 7,570,932 trainable parameters; the full model has 12,050,996. This is not a capacity-matched experiment.
- Initialization: reconstruct each seed's full model and verify its decoder/backbone hashes against the archived early-layer initialization. Copy shared tensors exactly. For the first three fusion convolutions, keep the leading input channels corresponding to the upsampled decoder and Semantic skip; remove only the Structural input channels, without rescaling the retained weights.
- Selection: select raw or EMA weights by validation AbsRel. Evaluate the new controls on the official test set only after all five control trainings complete. This ordering applies to the new controls, not to the earlier layer-selection study.

The names Semantic and Structural describe intended architectural roles. The original Early configuration lowers whole-image and primary-boundary test AbsRel in all five seeds. Mean whole-image error decreases from 0.1412 ± 0.0184 to 0.1283 ± 0.0113, a 9.2% relative reduction; paired differences are −0.0130 ± 0.0129. Middle and Late also lower whole-image error in all five seeds. Boundary improvement holds in five seeds for Middle and four for Late. These results support the full architecture under this recipe, without proving information disentanglement or an advantage independent of parameter count. Gains vary across seeds and are largest in the less stable control seeds 44 and 45.

## Verify and rebuild the completed comparison

```bash
python -m pip install -r requirements-analysis.txt
python -m scripts.component_study.verify
python -m scripts.component_study.build
```

These CPU-only commands check the archive hashes, all 20 validation minima, matching official-test identities, per-image aggregates, and the original paired summaries. They generate Markdown/LaTeX tables, a per-seed CSV, and full-precision JSON under `study/component-tables/`. No trained weights or dataset images are needed. The serialized checkpoints were checked against the histories by the on-server collector; the public verifier repeats the history/metadata checks without weights.

## Run

Use a separate checkout for each seed, with the dependencies, data, and pinned backbone described in [reproducibility.md](reproducibility.md). In each checkout, prepare its seed before starting any training. For example:

```bash
python -m scripts.layer_study.prepare --seed 42 \
  --train-root /path/to/nyu_depth_v2/sync \
  --test-root /path/to/nyu_depth_v2/official_splits/test --audit
CUDA_VISIBLE_DEVICES=0 python -m scripts.component_study.run --smoke
CUDA_VISIBLE_DEVICES=0 python -m scripts.component_study.run --phase train
```

Repeat for seeds 43–46 in their respective checkouts. Assign each running job its own GPU. Re-running the training command resumes from the last completed epoch; configuration hashes must match. The runner reuses the existing training loop rather than defining a separate optimizer or evaluator. Initialization is checked before training against the same-seed archived hashes; a mismatch fails rather than silently changing the comparison.

After all five trainings finish, run this command in each seed's checkout, replacing the paths with the five actual checkout roots:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.component_study.run --phase test \
  --control-roots /path/to/seed-42 /path/to/seed-43 /path/to/seed-44 \
                  /path/to/seed-45 /path/to/seed-46
```

The test gate requires five distinct seeds, complete 20-epoch histories, completion records, and checkpoints. Each checkpoint is loaded with the training configuration hash checked. Do not run two commands simultaneously in the same checkout.

## Outputs and compatibility

Full runs write to `study/runs/main_only/`; smoke runs write to `study/smoke/main_only/`. The stable machine identifier `main_only` means Semantic pathway + FiLM. Output files include `initialization_audit.json`, `initialization.json`, `effective_config.json`, `history.jsonl`, `training_complete.json`, `best.pt`, and `last.pt`. Official evaluation adds `test_metrics.json` and `test_per_image.jsonl` for the 654 test images.

The final comparison reports each seed, five-seed mean and sample SD, and paired full-minus-control differences. Every run is retained. Use `scripts.component_study.build` for the component comparison. The three-variant `scripts.multiseed.build` continues to describe the 15 layer-selection runs.

The release runner is a portable version of the isolated control experiment launcher. Local paths and descriptive configuration fields differ, so its configuration hashes are intentionally different. It is for fresh reproductions, not resuming checkpoints from that separate experiment directory. Existing experiment files and checkpoints should remain with their original launcher.
