# Configuration layout

- `layer_study/{early,middle,late}.yaml`: portable templates for the controlled v1 NYU comparison. Run `python -m scripts.layer_study.prepare --train-root PATH --test-root PATH --audit` to write local configurations under ignored `study/configs/`.
- `kitti.yaml`, `nyu.yaml`: historical release entry points, with the original evaluation-split selection behavior. They are not the controlled-study configurations.
- `nyu_siglip.yaml`: historical SigLIP backbone-transfer configuration.

Use the [reproduction guide](../docs/reproducibility.md) for the revised study and [legacy protocol notes](../docs/legacy_results.md) when interpreting earlier scores.

`main_path_indices` selects the FiLM-conditioned Semantic inputs. `structural_path_indices` selects the Structural inputs. Early is the original configuration; middle and late change those inputs without changing decoder size. The Semantic-only runner sets `use_structural_pathway=False` and preserves FiLM. See the [method guide](../docs/method.md) for the pathway/configuration mapping.
