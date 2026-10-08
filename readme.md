# TRM-CrossAttn 개선 요약

## 한눈에 보는 개선사항

### **핵심 아이디어**

**"문제(x)와 답(y)는 고정하고, 생각(z)만 반복하자!"**

## 프로젝트 개요

문제·답 표현과 반복 추론 상태를 분리하는 Cross-Attention 기반 Tiny Recursive Reasoning Model 실험입니다. Maze와 Sudoku 데이터 준비·학습 스크립트와 ARC 데이터/평가 도구를 포함합니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| [models/recursive_reasoning/trm_crossattn.py](models/recursive_reasoning/trm_crossattn.py) | 재귀 추론 모델 |
| [config/arch/trm_crossattn.yaml](config/arch/trm_crossattn.yaml) | 모델 구조 설정 |
| [config/cfg_pretrain.yaml](config/cfg_pretrain.yaml) | 학습 설정 |
| [dataset/](dataset/) | 데이터 생성 |
| [pretrain.py](pretrain.py) | 학습 진입점 |
| [evaluators/](evaluators/) | 평가 도구 |

## 데이터 준비와 학습

```bash
pip install -r requirements.txt
bash prepare_data.sh
bash train_sudoku.sh
```

`train_sudoku.sh`는 GPU 1개, `train_maze.sh`는 `torchrun`으로 GPU 4개를 사용하는 설정입니다. 데이터 경로·학습 기간·GPU 수를 실행 환경에 맞게 확인한 뒤 시작하세요.

기본 Cross-Attention 설정은 hidden size 512, attention heads 8, L layers 2입니다.
