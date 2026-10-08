<div align="center">

# TRACE

### Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation

**CoRL 2026 · 본회의 채택**

Zihao Li · Ranpeng Qiu · Yincong Chen · Guoqiang Ren · Weiming Zhi  
The University of Sydney · Zeno AI

[프로젝트](https://jeong-zju.github.io/trace/) · [빠른 시작](#빠른-시작) · [학습](../../docs/training.md) · [평가](../../docs/evaluation.md) · [인용](#인용)

[English](../../README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어**

</div>

![TRACE는 초기 단서가 사라진 뒤에도 이를 기억하여 나중의 의사결정에 활용합니다.](../../docs/assets/teaser.png)

**단서를 기억하고, 필요한 순간에 행동합니다.** TRACE는 초기 단서가 나중의 의사결정 전에
사라지는 작업을 위해 시각-운동 정책에 고정 크기의 인과적 메모리를 제공합니다.
경로 시그니처는 이미 실행한 로봇 상태 궤적을 요약하고, 시각·상태 정보의 기록과 검색을
라우팅합니다. 가벼운 어댑터를 통해 이 메모리를 ACT 및 Diffusion Policy와 연결합니다.

## 포함된 내용

- 회귀형(ACT) 및 확산형 인터페이스를 갖춘 **완전한 TRACE 주 방법**.
- 인과적 프리픽스 데이터셋, 시그니처 캐싱, 학습, 오프라인 및 시뮬레이션 평가.
- 합성 관측과 실제 경로 시그니처를 사용하며 사전 학습된 가중치를 다운로드하지 않는 CPU 예제.
- 참고 설정, 핵심 기능 테스트, 4개 언어 문서.

이 릴리스에는 주 방법만 포함됩니다. 구성 요소 제거 실험(ablation)의 구현, 비교 실험,
비공개 로봇 배포 설정은 제외됩니다. 데이터셋, 학습된 가중치, 논문의 개별 실행에 해당하는
전체 산출물은 **포함되어 있지 않습니다**. 참고 설정은 시작을 돕기 위한 것이며 논문 결과의
정확한 재현을 보장하지 않습니다. [공개 범위](../../docs/release-scope.md)를 확인하세요.

## 빠른 시작

Linux와 Python 3.12를 사용하세요. 압축을 푼 저장소의 루트에서 다음 명령을 실행합니다.
Signatory를 빌드하기 전에 시스템 FFmpeg와 C++ 컴파일러를 설치하세요.
GPU 학습에는 드라이버와 호환되는 PyTorch / CUDA 환경을 사용하세요.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e depends/signatory --no-build-isolation
python examples/smoke_trace.py
```

이 예제는 무작위로 초기화한 작은 CPU 모델을 사용합니다. 메모리를 거치는 전체 계산 경로가
유한한 행동 값을 출력하는지 확인하며, 작업 성공률을 측정하지는 않습니다.
환경 구성 및 빌드에 관한 자세한 내용은 [설치 안내](../../docs/installation.md)를 참고하세요.

## 자신의 데이터로 학습하기

동기화된 RGB 관측, 로봇 상태, 행동이 포함된 로컬 LeRobot v3 데이터셋을 준비하세요.
`data/local/task`를 데이터셋 디렉터리로 바꿉니다. 전처리는 별도의 출력 데이터셋을
생성하므로 입력 데이터셋도 계속 사용할 수 있습니다.

```bash
python data/process_dataset.py data/local/task \
  --output-dir data/local/task_trace --signature-depth 3

python scripts/train_policy.py --policy streaming_act \
  --defaults-path configs/trace_act.yaml --dataset data/local/task_trace

python scripts/train_policy.py --policy prism_diffusion \
  --defaults-path configs/trace_diffusion.yaml --dataset data/local/task_trace
```

두 설정 모두 TRACE의 전체 계산 경로와 고정된 난수 시드를 사용합니다.
실험 추적은 기본적으로 꺼져 있습니다. 데이터 분할, 정규화, 체크포인트에 관한 자세한 내용은
[데이터 준비](../../docs/data.md)와 [학습](../../docs/training.md)을 참고하세요.

## 체크포인트 평가하기

```bash
python scripts/eval_policy.py --policy streaming_act \
  --policy-path outputs/train/trace-act/RUN/checkpoints/STEP/pretrained_model \
  --dataset data/local/task_trace
```

체크포인트 경로를 실제 저장 디렉터리로 바꾸세요. 오프라인 행동 예측 오차는 로봇의 작업
성공률과 다릅니다. 시뮬레이션에는 해당 환경과 자산이 필요합니다.
[평가 안내](../../docs/evaluation.md)를 참고하세요.

## 코드 구성

| 경로 | 용도 |
| --- | --- |
| `policy/lerobot_policy_streaming_act/src/` | TRACE 메모리 및 ACT 인터페이스 |
| `policy/lerobot_policy_prism_diffusion/src/` | TRACE 메모리 및 Diffusion Policy 인터페이스 |
| `configs/` | 공개 주 방법의 학습 설정 |
| `scripts/` | 학습, 평가, 데이터셋 유틸리티 |
| `data/` | LeRobot 변환 및 경로 시그니처 전처리 |
| `examples/` | 실행 가능한 CPU 예제 |
| `tests/` | 주 방법, 데이터, 릴리스 검증 |
| `docs/` | 설치, 방법, 학습, 평가, 번역 |
| `depends/signatory/` | 함께 제공되는 시그니처 구현 및 원본 라이선스 |

`streaming_act`와 `prism_diffusion`은 두 TRACE 인터페이스에서 기존부터 직렬화에 사용한
정책 이름입니다. 주 방법 체크포인트의 파라미터 이름을 유지하기 위해 그대로 사용합니다.
이름 간의 대응 관계와 호환성 요구 사항은 [아키텍처](../../docs/architecture.md)를 참고하세요.

## 개발

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
```

모델 관련 의존성 없이 작은 CPU 유틸리티 테스트를 실행하려면 `requirements-test.txt`를
설치하고 `python -m pytest --lightweight`를 실행하세요.
개발 절차는 [CONTRIBUTING.md](../../CONTRIBUTING.md)를 참고하세요.

## 인용

```bibtex
@inproceedings{li2026trace,
  title     = {TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation},
  author    = {Li, Zihao and Qiu, Ranpeng and Chen, Yincong and Ren, Guoqiang and Zhi, Weiming},
  booktitle = {Conference on Robot Learning},
  year      = {2026},
  url       = {https://jeong-zju.github.io/trace/}
}
```

## 라이선스 및 감사의 말

TRACE 자체 기여분은 [Apache-2.0](../../LICENSE) 라이선스로 공개됩니다.
이 프로젝트는 LeRobot, ACT, Diffusion Policy, Signatory를 기반으로 합니다.
각 프로젝트의 고지와 라이선스는 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)에
보존되어 있습니다. 외부 데이터와 가중치에는 각각의 라이선스가 적용됩니다.
