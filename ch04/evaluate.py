"""04-2 실습: 학습 전후의 에이전트를 같은 조건으로 평가한다.

  uv run ch04/evaluate.py --model random                       # 무작위 행동
  uv run ch04/evaluate.py --model untrained                    # 학습 전 PPO (무작위 초기화된 신경망)
  uv run ch04/evaluate.py --model runs/ppo_cartpole/model.zip  # 학습된 PPO
"""
import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate(policy, episodes, seed):
    env = gym.make("CartPole-v1")
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return np.array(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/ppo_cartpole/model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000, help="학습에 쓰지 않은 시드로 평가")
    parser.add_argument("--init-seed", type=int, default=0, help="untrained 신경망의 초기화 시드")
    parser.add_argument("--stochastic", action="store_true", help="확률대로 행동을 뽑아 평가")
    args = parser.parse_args()

    if args.model == "random":
        rng = np.random.default_rng(args.seed)
        policy = lambda obs: int(rng.integers(2))
    else:
        if args.model == "untrained":
            model = PPO("MlpPolicy", "CartPole-v1", seed=args.init_seed)
        else:
            model = PPO.load(args.model)
        policy = lambda obs: int(model.predict(obs, deterministic=not args.stochastic)[0])

    returns = evaluate(policy, args.episodes, args.seed)
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"{args.model} ({mode}): {args.episodes}판 평균 {returns.mean():.1f} ± {returns.std():.1f} "
          f"(최소 {returns.min():.0f}, 최대 {returns.max():.0f}, 500점 만점 {np.sum(returns >= 500)}판)")


if __name__ == "__main__":
    main()
