# Config Layout

This directory is organized to keep GitHub-facing defaults clean.

- `kitti.yaml`: base KITTI config
- `nyu.yaml`: base NYU config
- The reported paper settings are folded into these base configs.

Legacy and intermediate experiment configs are moved to:

- `configs/legacy/`

Examples:

```bash
# Base configs
bash scripts/run_release_experiment.sh configs/kitti.yaml 0
bash scripts/run_release_experiment.sh configs/nyu.yaml 0
```
