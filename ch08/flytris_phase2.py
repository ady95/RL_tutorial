"""08-4 실습: FlyTris README의 2단계(Tetris 학습) 표를 재현한다.

FlyTris 커밋 8574ddc의 train_tetris.py는 기본 정책이 "button"(하행 뉴런이 버튼을 누름)인데,
README의 2단계 결과는 "cast"(좌우로 탐색하며 자리를 평가) 정책의 결과다. 명령행에는 정책을
바꾸는 옵션이 없으므로 이 스크립트로 cast 정책을 지정해 실행한다.

flytris 저장소의 가상환경에서 실행한다:
  cd flytris
  .venv/bin/python ../RL_tutorial/ch08/flytris_phase2.py --seeds 3 --episodes 200
  (Windows에서는 .venv/Scripts/python 을 쓴다)
"""
import argparse
import time

import numpy as np
from flytris.connectome.build import load_connectome
from flytris.experiments.train_tetris import TrainConfig, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--policy", default="cast", choices=["cast", "button", "gradient"])
    args = parser.parse_args()

    cfg = TrainConfig(episodes=args.episodes, policy=args.policy, device="cpu")
    connectome = load_connectome()
    rows = []
    for seed in range(args.seeds):
        start = time.time()
        print(f"\n  seed {seed}  (policy={args.policy})")
        r = train(seed=seed, cfg=cfg, connectome=connectome)
        rows.append(r)
        print(f"    학습 전 {r['pre']['lines'].mean():.2f}줄 | 학습 후 {r['post']['lines'].mean():.2f}줄 | "
              f"무작위 {r['random']['lines'].mean():.2f}줄  ({time.time() - start:.0f}초)")

    def col(key, field):
        return np.array([r[key][field].mean() for r in rows])

    print(f"\n  {'':<14}{'블록 수':>9}{'지운 줄':>9}{'구멍':>9}")
    for key, label in (("random", "무작위 배치"), ("pre", "학습 전 초파리"), ("post", "학습 후 초파리")):
        print(f"  {label:<14}{col(key, 'pieces').mean():9.1f}{col(key, 'lines').mean():9.2f}{col(key, 'holes').mean():9.1f}")


if __name__ == "__main__":
    main()
