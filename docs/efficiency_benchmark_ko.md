# SPACE-CLIP 효율 지표 실측 (GPU 0)

작성일: 2026-02-26  
대상 리포지토리: `SPACE-CLIP`  
벤치마크 스크립트: `scripts/benchmark_efficiency.py`

## 1. 목적

리뷰 대응 및 논문 보강을 위해 아래 항목을 정량화했다.

- 파라미터 수 (전체/학습 가능/백본/디코더)
- 추론 지연시간 (ms), FPS
- Peak GPU Memory
- 백본(CLIP Vision) 대비 전체 모델의 디코더 오버헤드

## 2. 측정 설정

- 장치: `cuda:0`
- 입력(CLIP): `224x224`
- 출력 해상도:
  - KITTI 설정: `352x704`
  - NYU 설정: `480x640`
- 측정 반복:
  - warmup: 10
  - measured iterations: 50
- 스크립트 실행 예:

```bash
.venv/bin/python scripts/benchmark_efficiency.py \
  --config configs/kitti.yaml \
  --gpu 0 \
  --batch-sizes 1 \
  --warmup 10 \
  --iters 50 \
  --out runs/release/efficiency_kitti_gpu0.json
```

## 3. 파라미터 수

두 설정(KITTI/NYU)에서 아키텍처가 동일하므로 값은 동일하다.

| 항목 | 값 |
|---|---:|
| Total params | 97,850,420 |
| Trainable params | 12,050,996 |
| Backbone total params | 85,799,424 |
| Backbone trainable params | 0 |
| Decoder total params | 12,050,996 |
| Decoder trainable params | 12,050,996 |

핵심: 학습은 디코더 파라미터만 진행되고, 비전 백본은 완전 고정(`b=0`)이다.

## 4. 핵심 실측 결과 (Batch=1)

소스:
- `runs/release/efficiency_kitti_gpu0.json`
- `runs/release/efficiency_nyu_gpu0.json`

| Config | Full latency (mean ms) | Full FPS | Full peak mem (MB) | CLIP-only latency (mean ms) | Decoder overhead (mean ms) |
|---|---:|---:|---:|---:|---:|
| KITTI (`configs/kitti.yaml`) | 5.253 | 190.35 | 465.30 | 3.362 | 1.891 |
| NYU (`configs/nyu.yaml`) | 5.238 | 190.91 | 465.30 | 3.364 | 1.874 |

요약:
- 배치 1 기준 전체 추론은 약 **5.24 ms/img**, 약 **190 FPS**
- 디코더 추가 오버헤드는 약 **1.87 ~ 1.89 ms**
- 추론 피크 메모리는 약 **465 MB**

## 5. 배치 스케일링 참고 (Batch=1,4)

소스:
- `runs/release/efficiency_kitti_gpu0_bs1_4.json`
- `runs/release/efficiency_nyu_gpu0_bs1_4.json`

### KITTI

| Batch | Full latency (mean ms) | Full FPS | Full peak mem (MB) |
|---|---:|---:|---:|
| 1 | 5.229 | 191.25 | 465.30 |
| 4 | 9.825 | 101.78 | 703.93 |

### NYU

| Batch | Full latency (mean ms) | Full FPS | Full peak mem (MB) |
|---|---:|---:|---:|
| 1 | 6.009 | 166.42 | 465.30 |
| 4 | 11.827 | 84.55 | 703.93 |

## 6. 해석 및 주의사항

- 본 실측 시점의 GPU는 다른 대형 프로세스들이 함께 점유 중이었다(공유 환경).
- 특히 NYU `batch=1,4`의 두 번째 측정 파일(`*_bs1_4.json`)에서 latency 분산이 커졌다.
  - 예: `full_model.latency_std_ms`가 상대적으로 큼.
- 따라서 논문 표에는 아래를 권장:
  - 배치 1 기준 수치를 메인으로 제시
  - `mean`과 함께 `p50/p90`을 병기하거나,
  - 가능한 경우 유휴 GPU에서 3회 반복 평균/표준편차 재측정

## 7. 논문 반영 문장 예시

- "Under the TFI-FB constraint, SPACE-CLIP runs at ~190 FPS (batch=1, RTX-class GPU) with ~465 MB peak inference memory."
- "The decoder adds only ~1.9 ms over the frozen CLIP vision forward pass, supporting efficient plug-in deployment."

