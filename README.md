# RL_tutorial

위키독스 책 《초파리 뇌로 배우는 강화학습 따라하기 — 16만 뉴런의 커넥톰으로 게임 AI 만들기》의 예제 코드 저장소입니다.

- 책: [https://wikidocs.net/book/21547](https://wikidocs.net/book/21547)

## 이 저장소로 만드는 것

1. CartPole + PPO — 일반 강화학습 완주
2. MaleCNS Connectome 분석 — pandas · NetworkX
3. nfly ConnectomeRNN + CartPole — 초파리 뇌 에이전트 학습
4. Fly Hero / FlyTris 따라하기 — 도파민 기반 생물학적 학습
5. 나만의 초파리 뇌 게임 — Random vs MLP vs Connectome 대결

## 빠른 시작

```bash
git clone https://github.com/ady95/RL_tutorial.git
cd RL_tutorial
uv sync                       # Python 3.12 가상환경과 패키지 설치
uv run ch02/check_env.py      # 실습 환경 점검
```

uv 설치 방법은 책의 02-1 또는 [uv 공식 문서](https://docs.astral.sh/uv/)를 참고하세요. 이 저장소는 Python 3.12 기준이며, Linux에서는 CPU용 PyTorch를 받도록 설정되어 있습니다.

## 폴더 구성

| 폴더 | 책의 장 | 내용 |
|---|---|---|
| ch02 | 02. 실습 환경 준비 | check_env.py 환경 점검, first_game.py 첫 CartPole |
| ch03 | 03. 게임으로 배우는 강화학습 | grid_world.py 직접 만든 Grid World, gym_api.py Gymnasium 구조, cartpole_random.py 무작위 기준선 |
| ch04 | 04. PPO로 게임 학습시키기 | train_ppo.py 학습, evaluate.py 평가, plot_rewards.py 학습 곡선, inspect_policy.py 정책망 분석 |
| ch05 | 05. Connectome을 Neural Network로 보기 | download_malecns.py 데이터 받기, explore_tables.py 표 탐색, neighbors.py 연결·경로 분석, to_matrix.py 가중치 행렬 |
| ch06 | 06. nfly로 초파리 Brain Agent 만들기 | inspect_agent.py 에이전트 조립·해부 (nfly 가상환경에서 uv run --project ../nfly 로 실행) |
| ch07 | 07. 초파리 뇌로 CartPole 학습시키기 | evaluate_nfly.py 체크포인트 평가 (nfly 가상환경), plot_curves.py 학습 곡선 비교. 체크포인트는 Releases의 ch07-cartpole-checkpoints |
| ch08 | 08. 초파리는 실제로 어떻게 학습할까 | mushroom_body.py 버섯체 회로 집계, flytris_phase2.py FlyTris 2단계(cast 정책) 재현 (flytris 가상환경) |
| ch09 | 09. 나만의 초파리 뇌 게임 만들기 | fly_forage.py 먹이 찾기 게임(FlyForage-v0), train_sb3.py·evaluate.py 일반 PPO, train_nfly.py·evaluate_nfly.py 초파리 뇌·MLP (nfly 가상환경) |
| ch10 | 10. 초파리 뇌 실험실과 남은 질문 | wiring_probe.py 배선 변형(뉴런 제거·섞기·무작위·부분 회로) 충분성 검사 (nfly 가상환경) |

장을 집필하는 대로 폴더가 추가됩니다.

## 필요 환경

- Windows / macOS / Linux, Python 3.12
- 1~5장은 CPU로 충분합니다. 커넥톰 전체를 학습하는 장은 GPU를 권장하며, CPU용 축소 설정을 함께 제공합니다.

## 데이터와 학습 결과물

커넥톰 데이터(`data/`), 학습 로그(`runs/`), 체크포인트(`checkpoints/`), 녹화 영상(`videos/`)은 용량이 커서 저장소에 포함하지 않습니다. 내려받는 방법과 생성 방법은 각 장의 본문을 참고하세요.

## 라이선스

이 저장소의 예제 코드는 [Apache License 2.0](LICENSE)을 따릅니다. 책 본문(위키독스)의 저작권은 저자에게 있습니다.
