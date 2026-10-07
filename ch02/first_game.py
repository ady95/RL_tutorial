"""02-3 실습: 무작위 행동으로 CartPole 한 판을 실행한다.

  uv run ch02/first_game.py            # 창을 띄워 직접 보기
  uv run ch02/first_game.py --save     # 창 없이 실행하고 마지막 장면을 PNG로 저장
"""
import argparse
from pathlib import Path

import gymnasium as gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="창 없이 실행하고 장면을 PNG로 저장")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("CartPole-v1", render_mode="rgb_array" if args.save else "human")
    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    total_reward, steps = 0.0, 0
    while True:
        action = env.action_space.sample()  # 0 = 왼쪽으로 밀기, 1 = 오른쪽으로 밀기
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"게임 종료: {steps}스텝 버팀, 총 보상 {total_reward:.0f}")
    print(f"마지막 관측값: {obs.round(3)}")

    if args.save:
        import matplotlib.pyplot as plt

        out = Path("outputs") / "ch02_first_game.png"
        out.parent.mkdir(exist_ok=True)
        plt.imsave(out, env.render())
        print(f"마지막 장면 저장: {out}")
    env.close()


if __name__ == "__main__":
    main()
