# Historical release and protocol differences

The original release advertised KITTI Eigen AbsRel **0.0901** and NYU Depth V2 AbsRel **0.1042**. Those experiments used the official evaluation splits during checkpoint selection. The exact evaluator revision for those two numbers was not preserved; an older release evaluator applied median scaling unconditionally. Consequently, these numbers should not be presented as directly comparable with the later validation-selected, unscaled controlled NYU study.

Separate historical NYU results (CLIP 0.1037 and SigLIP 0.1022) used median scaling and test-selected checkpoints. They are historical observations, not results from the new early/middle/late layer comparison. The legacy ablation's full-model recipe also differed from the other arms, so it does not isolate component synergy.

The original entry points remain available for compatibility:

```bash
bash scripts/run_release_experiment.sh configs/kitti.yaml 0
bash scripts/run_release_experiment.sh configs/nyu.yaml 0
python scripts/eval_spaceclip_checkpoint.py \
  --config configs/kitti.yaml \
  --checkpoint checkpoints/SPACE_CLIP_KITTI/best_checkpoint.pt \
  --crop eigen --median-scaling false
```

Current evaluator flags make crop/scaling explicit, but changing flags does not reconstruct the unknown historical evaluator or fix earlier checkpoint selection. Use `scripts/layer_study/` and `configs/layer_study/` for the controlled v1 study. Original qualitative illustrations in `figures/fig1.png`–`fig4.png` and the older release/efficiency logs are historical material.
