# NYU shared-encoder integration measurements

`nyu.json` is the retained three-configuration record used by the paper's integration-cost table. Its bytes match the archived `integration_benchmark_nyu_gpu1_subproc.json`; `checksums.json` records its SHA-256.

The record reports a batch-size-one comparison of a backbone alone, shared encoding with SPACE-CLIP, and a separate depth encoder. The recorded procedure uses isolated subprocesses, 10 warm-up iterations, and 50 timed iterations. Device `cuda:1` is recorded; the JSON does not identify the GPU model or software environment. No new measurement is implied by adding this record to the repository.

See [measurement notes](../../docs/efficiency.md) for the table and the distinction from the separately recorded full-model timing runs. This is an inference microbenchmark, not a downstream robot-control evaluation.
