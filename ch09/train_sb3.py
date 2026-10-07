"""09-3 실습: 일반 PPO(MLP) 에이전트로 FlyForage를 학습시켜 기준선을 만든다.

  uv run ch09/train_sb3.py --timesteps 100000 --seed 0
"""
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import fly_forage  # noqa: F401  (FlyForage-v0 등록)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", default="FlyForage-v0")
    args = parser.parse_args()

    name = "forage" if args.env == "FlyForage-v0" else args.env.removesuffix("-v0")
    run_dir = Path("runs") / f"{name}_sb3_s{args.seed}"
    # SB3는 렌더링 방식을 주지 않으면 rgb_array를 넣으므로, 이 게임이 지원하는 ansi를 준다
    env = make_vec_env(args.env, n_envs=4, seed=args.seed, monitor_dir=str(run_dir),
                       env_kwargs={"render_mode": "ansi"})
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0)
    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    model.save(run_dir / "model")
    print(f"학습 완료: {args.timesteps:,} 스텝, {time.time() - start:.0f}초 → {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
