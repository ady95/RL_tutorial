"""04-2 실습: Stable-Baselines3 PPO로 CartPole을 학습시킨다.

  uv run ch04/train_ppo.py                         # 기본 10만 스텝
  uv run ch04/train_ppo.py --lr 1e-3 --name lr1e-3 # 하이퍼파라미터를 바꿔 다른 이름으로 저장
"""
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="한 번 업데이트에 모으는 스텝 수(환경 1개당)")
    parser.add_argument("--n-envs", type=int, default=4, help="동시에 돌리는 환경 수")
    parser.add_argument("--name", default="ppo_cartpole")
    args = parser.parse_args()

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Monitor가 판마다 보상과 길이를 run_dir/*.monitor.csv에 기록한다
    env = make_vec_env("CartPole-v1", n_envs=args.n_envs, seed=args.seed, monitor_dir=str(run_dir))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        seed=args.seed,
        verbose=1,
        tensorboard_log="runs/tensorboard",
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name=args.name)
    elapsed = time.time() - start

    model.save(run_dir / "model")
    print(f"\n학습 완료: {args.timesteps:,} 스텝, {elapsed:.0f}초")
    print(f"모델 저장: {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
