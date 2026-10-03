# Five-seed Semantic-only control

This archive contains all five completed Semantic-pathway + FiLM controls (seeds 42–46). They use the controlled NYU recipe with the Structural pathway disabled. The experiment was added after the original layer-selection results were available. All five new trainings completed before any new control was evaluated on the official test set; all checkpoints were selected by validation AbsRel.

Each seed directory contains the 20-epoch training history, initialization and smoke-test audits, completion record, test summary, and 654 per-image test records. JSONL files are losslessly compressed. Images and trained weights are not redistributed. Original configuration identities remain in the run metadata; machine-local paths are omitted from the shared settings in `protocol.json`.

`component_comparison.json` contains every region and metric for the declared primary Early comparison and secondary Middle/Late comparisons, including seed-paired differences. The original dual-pathway records remain in `../nyu_multiseed`. `provenance.json` records the experimental source identities and environment. `checksums.json` covers every numerical archive file.

The original Early model lowers whole-image and primary-boundary AbsRel in all five seeds. Whole-image mean ± sample SD is 0.1283 ± 0.0113, compared with 0.1412 ± 0.0184 for the control. The paired full-minus-control difference is −0.0130 ± 0.0129. All seeds are retained, including the less stable control runs 44 and 45. The full and reduced models contain 12,050,996 and 7,570,932 trainable parameters; this is not a capacity-matched comparison.

From the repository root:

```bash
python -m scripts.component_study.verify
python -m scripts.component_study.build
```

See [the protocol and training instructions](../../docs/component_ablation.md). The CPU verifier reaggregates all published per-image results and checks validation selection. It does not re-run inference without weights; the original on-server collector additionally inspected the serialized selected checkpoints.
