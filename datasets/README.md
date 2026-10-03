## Dataset Setup

This repository uses a single recommended setup path:

```bash
bash scripts/setup_datasets.sh
```

This script downloads KITTI/NYU files, extracts them, and validates split paths.

### Custom data root

By default, data is created under `./datasets`. To use a different location:

```bash
bash scripts/setup_datasets.sh /path/to/data_root
```

The expected structure is:

```text
<data_root>/
  kitti_nyu/
    kitti/
    nyu_depth_v2/  (and symlink: nyu -> nyu_depth_v2)
  _downloads/
```

### Config path alignment

Default configs (`configs/kitti.yaml`, `configs/nyu.yaml`) expect:

- `./datasets/kitti_nyu/kitti/...`
- `./datasets/kitti_nyu/nyu_depth_v2/...`

If you use a custom data root, update dataset paths in your config accordingly.

## Controlled NYU layer study

The layer study requires distinct training and official-test roots:

```text
nyu_depth_v2/
  sync/
    bathroom_0001/...        # RGB JPEG and sync_depth PNG files
    ...
  official_splits/test/
    bathroom/rgb_00045.jpg
    bathroom/sync_depth_00045.png
    ...
```

Use the exact paths in `train_test_inputs/nyudepthv2_{train,test}_files_with_gt.txt` as the authority for filenames. NYU depth PNG values are interpreted in millimetres. Obtain data from the [NYU Depth V2 project](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and follow its usage terms. Data are not bundled in this repository.

The historical setup script may require network access and preparation of the official test files. Do not assume it has produced the complete controlled-study layout until the audit succeeds:

```bash
python -m scripts.layer_study.prepare \
  --train-root /path/to/nyu_depth_v2/sync \
  --test-root /path/to/nyu_depth_v2/official_splits/test \
  --audit
```

This generates train/validation scene splits from the original training list, preserves the official test list, and keeps machine-specific configuration out of Git. See the [reproduction guide](../docs/reproducibility.md).
