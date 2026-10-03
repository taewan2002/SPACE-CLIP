# Inference and integration measurements

## Shared-encoder integration benchmark

The paper's three-configuration comparison is archived in [`results/integration_cost/nyu.json`](../results/integration_cost/nyu.json). It uses `configs/nyu.yaml`, batch size one, 10 warm-up iterations, and 50 timed iterations, with each configuration measured in an isolated subprocess.

| Configuration | Total parameters | Trainable parameters | Duplicated backbone parameters | Peak memory (recorded MB field) | Latency (ms/image) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backbone only | 85,799,424 | 0 | 0 | 346.9 | 3.368 |
| Shared backbone + SPACE-CLIP | 97,850,420 | 12,050,996 | 0 | 468.1 | 5.200 |
| Separate depth backbone | 183,649,844 | 12,050,996 | 85,799,424 | 796.1 | 8.512 |

The field name and rounding follow the retained record and the manuscript. The record identifies `cuda:1`, but does not identify the GPU model or software environment; these numbers describe that measured setup. They do not measure robotic control or end-to-end VLA performance. Shared encoding assumes compatible backbone weights, preprocessing, and intermediate features.

The `benchmark_efficiency.py` command below measures full-model and CLIP-only timing. It is a different benchmark and does not regenerate this three-configuration comparison. The following measurements are kept separate so they are not substituted for the paper's integration table.

## Separate full-model timing records

The measurements below were recorded on 2026-02-26 with
`scripts/benchmark_efficiency.py`. They describe the historical release and
are separate from the controlled layer-selection study. The retained summary
identifies device `cuda:0` but does not specify the GPU model. Other workloads
were running on the device during measurement.

### Configuration

- CLIP input: 224 x 224.
- Output: 352 x 704 for KITTI and 480 x 640 for NYU.
- Warmup: 10 iterations; timed measurement: 50 iterations.
- Parameters: 97,850,420 total, with 85,799,424 frozen backbone parameters and
  12,050,996 trainable decoder parameters.

```bash
python scripts/benchmark_efficiency.py \
  --config configs/kitti.yaml --gpu 0 --batch-sizes 1 \
  --warmup 10 --iters 50 --out runs/efficiency_kitti.json
```

Repeat with `configs/nyu.yaml` for NYU. The full model and CLIP-only forward
passes are timed separately; their mean-latency difference is the reported
decoder overhead. Peak memory is allocated CUDA memory in MiB, although the
historical script names its output field `peak_memory_mb`.

### Batch size 1

| Configuration | Full latency (ms) | Images/s | Peak memory (MiB) | CLIP-only latency (ms) | Decoder overhead (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| KITTI | 5.253 | 190.35 | 465.30 | 3.362 | 1.891 |
| NYU | 5.238 | 190.91 | 465.30 | 3.364 | 1.874 |

Original record names were `efficiency_kitti_gpu0.json` and
`efficiency_nyu_gpu0.json` under the local `runs/release/` directory. Those
machine-local JSON files are not included in this repository.

### Additional batch-size measurements

| Configuration | Batch size | Full latency (ms/batch) | Batches/s | Peak memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| KITTI | 1 | 5.229 | 191.25 | 465.30 |
| KITTI | 4 | 9.825 | 101.78 | 703.93 |
| NYU | 1 | 6.009 | 166.42 | 465.30 |
| NYU | 4 | 11.827 | 84.55 | 703.93 |

The original files were `efficiency_kitti_gpu0_bs1_4.json` and
`efficiency_nyu_gpu0_bs1_4.json`. The script's `fps` field is `1000 / latency_ms`,
which counts batches per second. It equals images per second only at batch
size 1; multiply by batch size for image throughput.

The separate NYU timing runs show variation under shared-device load. Treat
these values as historical observations, not standardized comparisons with
other methods. New timing measurements should identify the GPU, software,
batch size, and concurrent load, and report repeated measurements.
