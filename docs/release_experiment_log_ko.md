# SPACE-CLIP 릴리즈 실험 로그 해설 (한국어)

문서 목적:
- `KITTI + NYU Depth V2` 릴리즈 실험 결과를 한국어로 정리
- 숫자 자체뿐 아니라 **왜 개선됐는지**, **논문에 어떻게 써야 하는지**까지 포함
- 향후 논문 리비전/재현 시 이 문서 하나로 전체 맥락을 복원

---

## 1. 한 줄 요약

- 현재 릴리즈 런은 동일 제약(TFI-FB)에서 `KITTI`와 `NYU` 모두 완료됨.
- 핵심 성능:
  - KITTI: `AbsRel 0.0901`
  - NYU: `AbsRel 0.1042` (best checkpoint 기준)
- 결론: 절대 SOTA 프레이밍보다는 **제약 조건 하에서 경쟁력 + 통합 용이성**을 강조하는 전략이 적합.

---

## 2. 실험 조건 정의 (핵심 용어)

### 2.1 TFI-FB 제약

- `TFI` (Text-Free Inference): 추론 시 텍스트 경로/텍스트 인코더를 사용하지 않음
- `FB` (Frozen Backbone): 비전 백본(CLIP vision encoder)을 학습 중 업데이트하지 않음

표기 관례:
- `t=0`: inference 시 text conditioning 없음
- `b=0`: training 시 backbone update 없음

즉, 본 실험의 핵심은
> **거대 백본은 고정하고(decoder-only adaptation), 텍스트 경로 없이 depth를 복원하는 설정**
입니다.

---

## 3. 최종 수치 정리

## 3.1 KITTI (TFI-FB, epoch 20 완료)

| Metric | Value |
|---|---:|
| abs_rel | 0.0901262663 |
| sq_rel | 0.4700706144 |
| rmse | 3.8450757744 |
| rmse_log | 0.1527595095 |
| log_10 | 0.0406452918 |
| a1 | 0.9087675153 |
| a2 | 0.9811781344 |
| a3 | 0.9944748649 |
| silog | 14.8046454564 |
| val_loss | 1.4335795678 |

해석:
- KITTI에서 개선 폭이 큼.
- `a2/a3`가 높아 전반적인 depth ordering은 안정적.
- `abs_rel` 하락이 가장 실질적인 개선 신호.

## 3.2 NYU Depth V2 (TFI-FB, epoch 20 완료)

| Metric | Best | Last |
|---|---:|---:|
| abs_rel | 0.1041774995 | 0.1050329669 |
| sq_rel | 0.0559715944 | 0.0567738119 |
| rmse | 0.3848016570 | 0.3884806082 |
| rmse_log | 0.1367771109 | 0.1379605836 |
| log_10 | 0.0446081604 | 0.0450048556 |
| a1 | 0.8958239667 | 0.8938787580 |
| a2 | 0.9838694890 | 0.9829804249 |
| a3 | 0.9973200294 | 0.9971803486 |
| silog | 12.8232017750 | 12.9645190017 |
| val_loss | 0.9907442254 | 1.0004627915 |

해석:
- NYU는 초반 빠르게 떨어지고 후반 plateau가 뚜렷함.
- Best와 Last 차이가 크지 않아, 과적합보다는 조기 수렴/정체 패턴에 가까움.

---

## 4. 학습 추세 해설

## 4.1 KITTI 추세

- 초반 selected AbsRel 약 `0.1953`
- 후반 `~0.092`까지 안정 하락
- 로그상 best selected AbsRel은 `0.0919` (17~19 epoch 부근)
- 최종 epoch selected AbsRel은 `0.0920`

의미:
- 설정이 불안정한 스파이크 없이 수렴함.
- 최종 구간에서 개선량이 작아지는 전형적인 안정 수렴 패턴.

## 4.2 NYU 추세

- epoch 1: `0.1384`
- epoch 9: `0.1060` 부근까지 급하락
- 이후 `0.1060~0.1074` 박스권

의미:
- “빠른 초반 학습 + 이른 정체” 패턴.
- 구조적 개선 없이 epoch만 늘리면 이득이 작다는 근거.

---

## 5. v1 -> v3(튜닝 레시피)에서 실제로 바뀐 것

중요:
- **코어 아키텍처 자체는 유지**됨.
  - frozen CLIP vision encoder
  - dual-pathway decoder (semantic + structural)
  - FiLM semantic conditioning

주요 변경은 학습/평가 프로토콜:
1. 스케줄러 변경: `step -> cosine_warmup`
2. multi-scale auxiliary SILog 활성화  
   `aux_loss_weights=[0.10, 0.05, 0.00]`
3. EMA 도입 및 best 선택 로직 강화  
   `ema_decay=0.996`, `eval_both_for_best=true`
4. TTA 정책 분리  
   학습 중 val TTA off, 최종 checkpoint 평가에서만 flip TTA on  
   (`eval_flip_tta=false`, `final_eval_flip_tta=true`)
5. KITTI 학습 길이 증가 (`10 -> 20`), NYU는 20 유지

핵심 메시지:
> 이 개선은 “새 백본/새 모델”이 아니라 **프로토콜 업그레이드**에 의한 성능 상승이다.

---

## 6. 무엇이 효과 있었고, 무엇이 한계였는가

## 6.1 효과 있었던 부분

- frozen-backbone + decoder-only 구조가 KITTI/NYU 모두에서 안정 학습
- EMA 기반 선택이 중후반 구간에서 유리
- auxiliary supervision이 초반 수렴 안정화에 기여
- 모듈형 통합 관점(VLA plugin)에서 구조적 장점 유지

## 6.2 아직 남은 한계

- 절대 성능은 task-specialized SOTA 대비 격차 존재
- NYU는 early plateau로 추가 개선 여지 제한
- 장시간 학습 대비 추가 이득이 작음

---

## 7. 논문 작성 전략 (실전)

## 7.1 권장 포지셔닝

- “absolute SOTA”가 아니라
- **strict constraint 하 경쟁력 + 통합성 + 유지보수성**을 강조

권장 표현:
- “competitive under strict constraints”
- “decoder-only spatial perception module”
- “text-free inference and frozen backbone reduce integration cost”

## 7.2 피해야 할 표현

- 범위를 명확히 하지 않은 과장 표현  
  예: “dramatically outperforms” (전체 SOTA 대비 의미로 읽힐 수 있음)

## 7.3 리뷰 대응 필수 포인트

1. validation/best checkpoint 선정 로직 명확화
2. KITTI+NYU cross-dataset 결과 본문 포함
3. 효율성 정량 지표 추가 (params/FLOPs/FPS/memory)
4. qualitative에서 성공/실패 사례를 분리 설명

---

## 8. 다음 실험 우선순위

1. 효율성 테이블 추가  
   (동일 하드웨어/동일 배치에서 2~3 baseline 비교)
2. NYU seed 반복 (`n>=3`)로 분산 보고
3. NYU plateau 원인 분석  
   - early stop
   - aux loss weight 미세조정
   - augmentation 강화

---

## 9. 운영 팁

- detached `screen` 프로세스가 남아도, 결과 진실은 run 디렉토리의 metrics 파일 기준으로 판단
- 체크포인트별 추세는 `train.log`와 `best_metrics*.json`을 함께 확인

---

## 10. 빠른 점검 명령어

```bash
screen -ls
ps -eo pid,etimes,cmd --sort=-etimes | rg "run_release_experiment.sh|eval_spaceclip_checkpoint.py|kitti|nyu"
tail -f runs/release/kitti_nono_1235_20260224_132035/train.log
tail -f runs/release/nyu_nono_1235_20260224_164007/train.log
```

---

## 11. 최종 판단

- 현재 결과로 논문 작성은 가능.
- 단, 메시지의 중심은 아래로 고정하는 것이 안전:
  1. **TFI-FB 제약에서의 성능**
  2. **모듈형 통합 이점**
  3. **재현 가능한 학습 프로토콜 업그레이드**

