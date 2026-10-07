"""09-3, 09-5 실습: FlyForage에서 에이전트를 같은 조건으로 평가한다.

  uv run ch09/evaluate.py --agent random
  uv run ch09/evaluate.py --agent sb3 --model runs/forage_sb3_s0/model.zip
  uv run ch09/evaluate.py --agent sb3 --model runs/forage_sb3_s0/model.zip --env FlyForageWide-v0
  uv run ch09/evaluate.py --agent sb3 --model runs/forage_sb3_s0/model.zip --noise 0.2
"""
import argparse

import gymnasium as gym
import numpy as np

import fly_forage  # noqa: F401


def run(policy, env_id, episodes, seed, noise):
    env = gym.make(env_id, obs_noise=noise)
    rets, foods, lengths, crashes = [], [], [], 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            if terminated or truncated:
                break
        rets.append(total)
        foods.append(info["food_eaten"])
        lengths.append(env.unwrapped.steps)
        crashes += int(terminated)
    return np.array(rets), np.array(foods), np.array(lengths), crashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "sb3"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--env", default="FlyForage-v0")
    parser.add_argument("--noise", type=float, default=0.0, help="관측에 더할 가우시안 잡음의 표준편차")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if args.agent == "random":
        rng = np.random.default_rng(args.seed)
        policy = lambda obs: int(rng.integers(3))
    else:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        policy = lambda obs: int(model.predict(obs, deterministic=True)[0])

    rets, foods, lengths, crashes = run(policy, args.env, args.episodes, args.seed, args.noise)
    print(f"{args.agent} @ {args.env} noise={args.noise}: 평균 보상 {rets.mean():.2f} ± {rets.std():.2f} | "
          f"먹이 {foods.mean():.1f}개 | 버틴 스텝 {lengths.mean():.0f} | 끝까지 생존 {args.episodes - crashes}/{args.episodes}판")


if __name__ == "__main__":
    main()
