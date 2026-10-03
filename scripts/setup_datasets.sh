#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${1:-$WORKDIR/datasets}"
BASE_DIR="$DATA_ROOT/kitti_nyu"
DL_DIR="$DATA_ROOT/_downloads"
KITTI_DL_DIR="$DL_DIR/kitti"
NYU_DL_DIR="$DL_DIR/nyu"

mkdir -p "$BASE_DIR" "$KITTI_DL_DIR" "$NYU_DL_DIR"

log() {
  printf '[setup-datasets] %s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"

  if have_cmd aria2c; then
    aria2c -x 8 -s 8 -k 1M -c --auto-file-renaming=false -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  else
    wget -c -O "$out" "$url"
  fi
}

# 0) Ensure NYU split files exist in train_test_inputs
NYU_TRAIN_SPLIT="$WORKDIR/train_test_inputs/nyudepthv2_train_files_with_gt.txt"
NYU_TEST_SPLIT="$WORKDIR/train_test_inputs/nyudepthv2_test_files_with_gt.txt"
if [ ! -f "$NYU_TRAIN_SPLIT" ] || [ ! -f "$NYU_TEST_SPLIT" ]; then
  log "downloading NYU split files into train_test_inputs"
  download_file "https://raw.githubusercontent.com/cleinc/bts/master/train_test_inputs/nyudepthv2_train_files_with_gt.txt" "$NYU_TRAIN_SPLIT"
  download_file "https://raw.githubusercontent.com/cleinc/bts/master/train_test_inputs/nyudepthv2_test_files_with_gt.txt" "$NYU_TEST_SPLIT"
fi

# 1) Build required KITTI sequence/date list from existing split files
KITTI_SEQ_LIST="$KITTI_DL_DIR/kitti_sequences.txt"
KITTI_DATE_LIST="$KITTI_DL_DIR/kitti_dates.txt"
python - <<PY
from pathlib import Path

workdir = Path("$WORKDIR")
seqs = set()
dates = set()
for fp in [workdir / "train_test_inputs/kitti_eigen_train_files_with_gt.txt", workdir / "train_test_inputs/kitti_eigen_test_files_with_gt.txt"]:
    for line in fp.read_text().splitlines():
        if not line.strip():
            continue
        img = line.split()[0]
        parts = img.split('/')
        if len(parts) < 2:
            continue
        date = parts[0]
        drive_sync = parts[1]
        if not drive_sync.endswith('_sync'):
            continue
        dates.add(date)
        seqs.add(drive_sync.replace('_sync', ''))

(Path("$KITTI_SEQ_LIST")).write_text("\n".join(sorted(seqs)) + "\n")
(Path("$KITTI_DATE_LIST")).write_text("\n".join(sorted(dates)) + "\n")
print(f"sequences={len(seqs)} dates={len(dates)}")
PY

# 2) Download KITTI depth gt and raw sequences
log "downloading KITTI data_depth_annotated.zip"
download_file "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip" "$KITTI_DL_DIR/data_depth_annotated.zip"

log "downloading KITTI calibration files"
if have_cmd aria2c; then
  KITTI_CALIB_URLS="$KITTI_DL_DIR/kitti_calib_urls.txt"
  : > "$KITTI_CALIB_URLS"
  while IFS= read -r date; do
    [ -z "$date" ] && continue
    printf 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/%s_calib.zip\n out=%s_calib.zip\n' "$date" "$date" >> "$KITTI_CALIB_URLS"
  done < "$KITTI_DATE_LIST"
  aria2c -c -j 3 -x 8 -s 8 -k 1M --auto-file-renaming=false -d "$KITTI_DL_DIR" -i "$KITTI_CALIB_URLS"
else
  while IFS= read -r date; do
    [ -z "$date" ] && continue
    download_file "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}_calib.zip" "$KITTI_DL_DIR/${date}_calib.zip"
  done < "$KITTI_DATE_LIST"
fi

log "downloading KITTI raw sync sequence zips"
if have_cmd aria2c; then
  KITTI_RAW_URLS="$KITTI_DL_DIR/kitti_raw_urls.txt"
  : > "$KITTI_RAW_URLS"
  while IFS= read -r seq; do
    [ -z "$seq" ] && continue
    printf 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/%s/%s_sync.zip\n out=%s_sync.zip\n' "$seq" "$seq" "$seq" >> "$KITTI_RAW_URLS"
  done < "$KITTI_SEQ_LIST"
  aria2c -c -j 8 -x 8 -s 8 -k 1M --auto-file-renaming=false -d "$KITTI_DL_DIR" -i "$KITTI_RAW_URLS"
else
  while IFS= read -r seq; do
    [ -z "$seq" ] && continue
    download_file "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${seq}/${seq}_sync.zip" "$KITTI_DL_DIR/${seq}_sync.zip"
  done < "$KITTI_SEQ_LIST"
fi

# 3) Extract KITTI archives
mkdir -p "$BASE_DIR/kitti"
log "extracting KITTI depth zip"
unzip -oq "$KITTI_DL_DIR/data_depth_annotated.zip" -d "$BASE_DIR/kitti"

log "extracting KITTI calibration zips"
while IFS= read -r date; do
  [ -z "$date" ] && continue
  unzip -oq "$KITTI_DL_DIR/${date}_calib.zip" -d "$BASE_DIR/kitti"
done < "$KITTI_DATE_LIST"

log "extracting KITTI raw sync zips"
while IFS= read -r seq; do
  [ -z "$seq" ] && continue
  unzip -oq "$KITTI_DL_DIR/${seq}_sync.zip" -d "$BASE_DIR/kitti"
done < "$KITTI_SEQ_LIST"

# 3.5) Reconcile KITTI depth paths for Eigen split
# Some Eigen train/test entries may point to depth files that exist in the opposite
# split folder (train/val). Mirror missing files with symlinks (fallback: copy)
# so the configured gt_path / gt_path_eval resolve consistently.
log "reconciling KITTI depth paths for Eigen split"
python - <<PY
from pathlib import Path
import os
import shutil

workdir = Path("$WORKDIR")
kitti_root = Path("$BASE_DIR") / "kitti"

train_split = workdir / "train_test_inputs/kitti_eigen_train_files_with_gt.txt"
test_split = workdir / "train_test_inputs/kitti_eigen_test_files_with_gt.txt"

def ensure_depth_files(split_file: Path, target_root: Path, fallback_root: Path):
    fixed = 0
    missing = 0
    for line in split_file.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        depth_rel = parts[1] if len(parts) > 1 else "None"
        if depth_rel == "None":
            continue

        target = target_root / depth_rel
        if target.exists():
            continue

        fallback = fallback_root / depth_rel
        if not fallback.exists():
            missing += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            rel = os.path.relpath(fallback, start=target.parent)
            os.symlink(rel, target)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(fallback, target)
        fixed += 1

    return fixed, missing

fixed_train, missing_train = ensure_depth_files(
    train_split, kitti_root / "train", kitti_root / "val"
)
fixed_test, missing_test = ensure_depth_files(
    test_split, kitti_root / "val", kitti_root / "train"
)

print(f"KITTI depth reconcile (train split): fixed={fixed_train}, still_missing={missing_train}")
print(f"KITTI depth reconcile (test split): fixed={fixed_test}, still_missing={missing_test}")
PY

# 4) Download + extract NYU (public preprocessed package)
log "downloading NYU package"
NYU_ZIP="$NYU_DL_DIR/nyu_depth_v2.zip"
download_file "https://huggingface.co/datasets/aradhye/nyu_depth_v2/resolve/main/nyu_depth_v2.zip" "$NYU_ZIP"

log "extracting NYU package"
unzip -oq "$NYU_ZIP" -d "$BASE_DIR"

# 5) Normalize NYU root path used by configs/scripts
if [ -d "$BASE_DIR/nyu_depth_v2" ] && [ ! -e "$BASE_DIR/nyu" ]; then
  ln -s "nyu_depth_v2" "$BASE_DIR/nyu"
fi

# 6) Validate key file existence for KITTI/NYU against split files
python - <<PY
from pathlib import Path

workdir = Path("$WORKDIR")
base = Path("$BASE_DIR")

kitti_root = base / "kitti"
kitti_gt_root_train = kitti_root / "train"
kitti_gt_root_val = kitti_root / "val"

kitti_train = workdir / "train_test_inputs/kitti_eigen_train_files_with_gt.txt"
kitti_test = workdir / "train_test_inputs/kitti_eigen_test_files_with_gt.txt"

missing_kitti_img = 0
missing_kitti_depth = 0
for fp, gt_root in [(kitti_train, kitti_gt_root_train), (kitti_test, kitti_gt_root_val)]:
    for line in fp.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        img_rel = parts[0]
        depth_rel = parts[1] if len(parts) > 1 else 'None'

        if not (kitti_root / img_rel).exists():
            missing_kitti_img += 1

        if depth_rel != 'None' and not (gt_root / depth_rel).exists():
            missing_kitti_depth += 1

nyu_candidates = [base / "nyu", base / "nyu_depth_v2"]
nyu_root = None
for c in nyu_candidates:
    if c.exists():
        nyu_root = c
        break

missing_nyu_img = None
missing_nyu_depth = None
if nyu_root is not None:
    nyu_train = workdir / "train_test_inputs/nyudepthv2_train_files_with_gt.txt"
    nyu_test = workdir / "train_test_inputs/nyudepthv2_test_files_with_gt.txt"

    missing_nyu_img = 0
    missing_nyu_depth = 0
    for fp in [nyu_train, nyu_test]:
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            # NOTE: this heredoc is unquoted, so '\\' must be written as '\\\\'
            # to survive shell processing and reach Python as a single backslash.
            img_rel = parts[0].lstrip('/').lstrip('\\\\')
            depth_rel = parts[1].lstrip('/').lstrip('\\\\') if len(parts) > 1 else None
            if not (nyu_root / img_rel).exists():
                missing_nyu_img += 1
            if depth_rel is not None and not (nyu_root / depth_rel).exists():
                missing_nyu_depth += 1

print('KITTI missing image files:', missing_kitti_img)
print('KITTI missing depth files:', missing_kitti_depth)
print('NYU root:', str(nyu_root) if nyu_root else 'NOT_FOUND')
print('NYU missing image files:', missing_nyu_img)
print('NYU missing depth files:', missing_nyu_depth)
PY

log "dataset setup completed"
