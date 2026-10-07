"""07-3 실습: nfly 체크포인트에 저장된 학습 기록으로 학습 곡선을 그리고 비교한다.

체크포인트(.pt)에는 학습 중 끝난 판들의 리턴 목록이 들어 있다. 이 스크립트는 nfly 없이
RL_tutorial 환경만으로 실행된다 (GPU가 없어도 내려받은 체크포인트로 분석 가능).

  uv run ch07/plot_curves.py --fly runs/nfly/fly-cp-s*.pt --mlp runs/nfly/mlp-cp-s*.pt
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_returns(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(payload["returns"], dtype=float)


def curve(returns, window):
    """CartPole은 스텝마다 보상 1이므로 리턴 누적합 = 누적 스텝 수."""
    steps = np.cumsum(returns)
    smooth = np.array([returns[max(0, i - window + 1):i + 1].mean() for i in range(len(returns))])
    return steps, smooth


def summarize(name, returns, window):
    steps, smooth = curve(returns, window)
    print(f"  {name:<16} {len(returns):>5}판 {int(steps[-1]):>7,}스텝 | 마지막 {window}판 평균 {returns[-window:].mean():6.1f} "
          f"| 이동평균 최고 {smooth.max():6.1f} | 이동평균이 40 미만인 판 비율 {np.mean(smooth < 40):5.1%}")
    return steps, smooth


def expand(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fly", nargs="*", default=[], help="초파리 뇌 체크포인트 (와일드카드 가능)")
    parser.add_argument("--mlp", nargs="*", default=[], help="같은 학습기로 학습한 MLP 체크포인트")
    parser.add_argument("--window", type=int, default=20, help="이동 평균 창 (nfly 로그의 mean return(20)과 같은 값)")
    parser.add_argument("--out", default="outputs/ch07_fly_vs_mlp.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, files, color in [("fly", expand(args.fly), "#C44E52"), ("mlp", expand(args.mlp), "#4C72B0")]:
        if files:
            print(f"[{label}]")
        for i, f in enumerate(files):
            steps, smooth = summarize(Path(f).stem, load_returns(f), args.window)
            ax.plot(steps, smooth, color=color, alpha=0.8, linewidth=1.2,
                    label=f"{label} ({len(files)} seeds)" if i == 0 else None)

    ax.axhline(23.7, color="gray", linestyle="--", linewidth=1, label="random (03-4)")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"episode return (moving avg {args.window})")
    ax.set_title("nfly simple PPO on CartPole-v1: fly connectome vs MLP")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"그래프 저장: {args.out}")


if __name__ == "__main__":
    main()
