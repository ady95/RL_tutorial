"""04-3 실습: Monitor 기록으로 Reward curve를 그린다. 여러 실행을 한 그래프에 겹쳐 그릴 수 있다.

  uv run ch04/plot_rewards.py runs/ppo_cartpole
  uv run ch04/plot_rewards.py runs/ppo_cartpole runs/lr1e-3 runs/lr3e-5
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_monitor(run_dir):
    frames = []
    for csv in sorted(Path(run_dir).glob("*.monitor.csv")):
        frames.append(pd.read_csv(csv, skiprows=1))  # 첫 줄은 메타데이터
    df = pd.concat(frames).sort_values("t")  # t: 학습 시작 후 경과 시간(초)
    df["timesteps"] = df["l"].cumsum()       # l: 판 길이, r: 판 보상
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--window", type=int, default=50, help="이동 평균 창 크기(판)")
    parser.add_argument("--out", default="outputs/ch04_reward_curve.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in args.runs:
        df = load_monitor(run)
        smooth = df["r"].rolling(args.window, min_periods=1).mean()
        ax.plot(df["timesteps"], smooth, label=Path(run).name)
        last = df["r"].tail(args.window).mean()
        print(f"{Path(run).name}: {len(df)}판, 마지막 {args.window}판 평균 보상 {last:.1f}")

    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"episode return (moving avg {args.window})")
    ax.set_title("PPO on CartPole-v1")
    ax.axhline(500, color="gray", linestyle=":", linewidth=1)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"그래프 저장: {args.out}")


if __name__ == "__main__":
    main()
