# Configuration layout

- `layer_study/{early,middle,late}.yaml`: portable templates for the controlled v1 NYU comparison. Run `python -m scripts.layer_study.prepare --train-root PATH --test-root PATH --audit` to write local configurations under ignored `study/configs/`.
- `kitti.yaml`, `nyu.yaml`: historical release entry points, with the original evaluation-split selection behavior. They are not the controlled-study configurations.
- `legacy/`: earlier experimental configurations retained for provenance.

Use the [reproduction guide](../docs/reproducibility.md) for the revised study and [legacy protocol notes](../docs/legacy_results.md) when interpreting earlier scores.
