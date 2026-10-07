"""03-4 실습: Random Agent로 CartPole을 여러 판 실행하고 통계를 낸다.

  uv run ch03/cartpole_random.py --episodes 100
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    print("Observation Space:", env.observation_space)
    print("Action Space:     ", env.action_space)

    env.action_space.seed(args.seed)
    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    env.close()

    returns = np.array(returns)
    print(f"\nRandom Agent {args.episodes}판")
    print(f"  평균 {returns.mean():.1f}  표준편차 {returns.std():.1f}")
    print(f"  최소 {returns.min():.0f}  중앙값 {np.median(returns):.0f}  최대 {returns.max():.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path("outputs") / "ch03_cartpole_random.png"
    out.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(returns, bins=20, color="#4C72B0")
    ax.axvline(returns.mean(), color="#C44E52", linestyle="--", label=f"mean {returns.mean():.1f}")
    ax.set_xlabel("episode return (steps survived)")
    ax.set_ylabel("episodes")
    ax.set_title("CartPole-v1, random agent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"히스토그램 저장: {out}")


if __name__ == "__main__":
    main()
