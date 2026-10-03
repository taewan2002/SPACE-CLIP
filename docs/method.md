# Dual-pathway depth decoding

SPACE-CLIP predicts monocular depth by adapting a decoder around a frozen vision encoder. The paper uses text-free inference with a frozen backbone (TFI-FB): all learned depth-specific parameters belong to the Dense Predictor, and inference does not require text prompts or a text encoder.

## Feature processing

The default backbone is CLIP ViT-B/16 with a bicubically resized 224 × 224 input. `L0` is the embedding output; `L1`–`L12` are transformer-block outputs. Every selected state supplies a 14 × 14 patch grid. The decoder increases spatial resolution; changing the encoder layer alone does not change patch-grid resolution.

| Part | Inputs and processing | Configuration / implementation |
| --- | --- | --- |
| Semantic pathway | States `[12, 9, 6, 3]`; pooled global image context generates per-channel FiLM scale and shift | `main_path_indices`, `use_film`, `film_param_generators`, `main_path_projections` |
| Structural pathway | Default states `[2, 1, 0]`; independent projections and spatial refinement without FiLM | `structural_path_indices`, `use_structural_pathway`, `StructuralPathwayBlock` |
| Fusion | Upsampling, gated skip features, concatenation, and learned convolutions; Structural features enter the first three stages | `MainDecoderBlock`, `main_decoder_blocks` |
| Prediction | Decoder stages reach 28, 56, 112, and 224 pixels per side; the final prediction is resized to the target dimensions | `decoder_channels`, `depth_prediction_head`, `output_size` |

The model is implemented in [`space_clip.py`](../space_clip.py). The default channel sequence is `[256, 128, 64, 32]`. Semantic and Structural describe the branches' processing roles in context-conditioned decoding and spatial refinement. They do not assert exclusive semantic/geometric information content.

The main loss combines SILog and SSIM, with auxiliary SILog supervision at the first two decoder outputs. Exact training, masking, and checkpoint-selection settings are in the [reproduction guide](reproducibility.md).

## Layer selection within the same architecture

| Configuration | Semantic indices | Structural indices | Trainable parameters |
| --- | --- | --- | ---: |
| Semantic-only + FiLM | `[12, 9, 6, 3]` | Disabled | 7,570,932 |
| SPACE-CLIP, early | `[12, 9, 6, 3]` | `[2, 1, 0]` | 12,050,996 |
| SPACE-CLIP, middle | `[12, 9, 6, 3]` | `[7, 5, 4]` | 12,050,996 |
| SPACE-CLIP, late | `[12, 9, 6, 3]` | `[11, 10, 8]` | 12,050,996 |

Early is the original configuration illustrated in the architecture figure. Middle and late are alternative inputs to the same Structural pathway. The controlled NYU results favor middle inputs in mean error and repeatability under the tested recipe; early is not claimed to be the optimal layer choice. The layer-selection study holds decoder size, initialization, data order, and training settings fixed within each seed.

The Semantic-only control retains FiLM. Its runner copies shared initial weights from the corresponding full model and removes Structural input channels from the affected fusion convolutions. It evaluates the contribution of the full architecture against a smaller decoder; it does not isolate architecture from parameter count.

## What the experiments establish

| Evidence | Question addressed | Scope |
| --- | --- | --- |
| Original NYU and KITTI benchmarks and qualitative figures | Can this decoder produce indoor and outdoor depth maps? | Original selection/evaluation conditions; see [legacy records](legacy_results.md) |
| CLIP/SigLIP transfer | Can the design be trained with another frozen backbone family? | Separately trained decoder configurations; not zero-shot decoder-weight transfer |
| Shared/separate encoder cost | What duplication cost does encoder reuse avoid? | Recorded inference microbenchmark, not an end-to-end robot-policy result |
| Five-seed Semantic-only comparison | Does the full model improve on Semantic-only + FiLM? | Whole-image and primary-boundary error improve in all five early/control pairs; parameter counts differ |
| Matched early/middle/late comparison | Which Structural inputs work well under the same recipe? | One frozen CLIP backbone and the controlled NYU protocol |
| Frozen-feature spectra | How does spatial-frequency content vary across encoder depth? | Feature-space measurements, including window and uniform-input controls |
| Structural-input frequency interventions | Does changing selected frequency components affect depth predictions? | Trained-model sensitivity on validation scenes, including energy-matched controls |

The new component comparison keeps FiLM enabled in both models. It does not separately establish FiLM's contribution or FiLM–Structural synergy. Likewise, frequency interventions do not prove that the two pathways contain disjoint information, or establish superiority of hierarchical fusion over a capacity-matched alternative.

See [component ablation](component_ablation.md), [five-seed aggregation](multiseed.md), and [Fourier reproduction](reproducibility.md#fourier-analysis) for the corresponding records and commands.

## Reuse and experiment entry points

Encoder reuse requires the host system to provide compatible weights, preprocessing, and intermediate states. The architecture motivates shared perception; downstream robotic control and VLA policy improvement remain outside the reported evaluation.

- [`configs/layer_study/`](../configs/layer_study/): portable early/middle/late templates for the controlled NYU experiments.
- [`scripts/component_study/`](../scripts/component_study/): Semantic-only control with paired initialization and validation-based checkpoint selection.
- [`configs/nyu.yaml`](../configs/nyu.yaml), [`configs/kitti.yaml`](../configs/kitti.yaml), and [`configs/nyu_siglip.yaml`](../configs/nyu_siglip.yaml): original benchmark entry points with the [legacy protocol](legacy_results.md).

Prepare fresh local paths and splits with the controlled-study setup command before training. Keep existing run configurations intact when resuming an experiment.
