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
