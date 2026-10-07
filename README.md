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
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

패키지 목록(`requirements.txt`)과 환경 점검 스크립트는 실측으로 기준 버전을 확정한 뒤 추가됩니다.

## 폴더 구성

| 폴더 | 책의 장 | 내용 |
|---|---|---|

장을 집필하는 대로 폴더가 추가됩니다.

## 필요 환경

- Windows / macOS / Linux, Python 3.11 이상
- 1~5장은 CPU로 충분합니다. 커넥톰 전체를 학습하는 장은 GPU를 권장하며, CPU용 축소 설정을 함께 제공합니다.

## 데이터와 학습 결과물

커넥톰 데이터(`data/`), 학습 로그(`runs/`), 체크포인트(`checkpoints/`), 녹화 영상(`videos/`)은 용량이 커서 저장소에 포함하지 않습니다. 내려받는 방법과 생성 방법은 각 장의 본문을 참고하세요.
