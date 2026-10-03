# Historical KITTI and NYU training records

These measurements come from the original release, before the controlled
five-seed study. They preserve the recorded values, including loss values and
checkpoint comparisons. The runs used official evaluation splits during
checkpoint selection, and the exact evaluator revision was not preserved.
See [protocol differences](legacy_results.md) before comparing these results
with the validation-selected study.

The released summaries rounded AbsRel to 0.0901 for KITTI and 0.1042 for NYU.

## KITTI

The historical run completed 20 epochs with a frozen CLIP backbone and no text
encoder at inference.

| Metric | Value |
| --- | ---: |
| AbsRel | 0.0901262663 |
| SqRel | 0.4700706144 |
| RMSE | 3.8450757744 |
| RMSE log | 0.1527595095 |
| Log10 | 0.0406452918 |
| Delta 1 | 0.9087675153 |
| Delta 2 | 0.9811781344 |
| Delta 3 | 0.9944748649 |
| SILog | 14.8046454564 |
| Evaluation loss (`val_loss`) | 1.4335795678 |

The training-log summary recorded selected AbsRel decreasing from about 0.1953
to 0.092. Its best selected value was approximately 0.0919 at epochs 17-19,
and its final-epoch value was approximately 0.0920. Those log values differ
from the final reported evaluation above; the archived summary does not
resolve that difference.

## NYU Depth V2

The historical run completed 20 epochs. The columns reproduce its selected
and final checkpoint reports; "best" does not indicate an independent
validation split in this historical experiment.

| Metric | Best | Last |
| --- | ---: | ---: |
| AbsRel | 0.1041774995 | 0.1050329669 |
| SqRel | 0.0559715944 | 0.0567738119 |
| RMSE | 0.3848016570 | 0.3884806082 |
| RMSE log | 0.1367771109 | 0.1379605836 |
| Log10 | 0.0446081604 | 0.0450048556 |
| Delta 1 | 0.8958239667 | 0.8938787580 |
| Delta 2 | 0.9838694890 | 0.9829804249 |
| Delta 3 | 0.9973200294 | 0.9971803486 |
| SILog | 12.8232017750 | 12.9645190017 |
| Evaluation loss (`val_loss`) | 0.9907442254 | 1.0004627915 |

The log summary recorded AbsRel of 0.1384 at epoch 1 and about 0.1060 at
epoch 9, followed by values around 0.1060-0.1074.

## Historical recipe changes

The frozen backbone and dual-pathway decoder were retained. The release
recipe changed the learning-rate schedule from step decay to cosine warmup,
added auxiliary SILog weights `[0.10, 0.05, 0.00]`, used EMA decay `0.996`,
and compared raw and EMA checkpoints. Flip averaging was disabled during
training-time evaluation and enabled for final checkpoint evaluation.
KITTI training increased from 10 to 20 epochs; NYU remained at 20 epochs.
These settings changed together, so the records do not isolate the effect
of any one change.

The controlled protocol and full five-seed measurements are documented in
[reproducibility.md](reproducibility.md) and [multiseed.md](multiseed.md).
