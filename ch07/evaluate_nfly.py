"""07-3 실습: nfly로 학습한 체크포인트(초파리 뇌 또는 MLP)를 같은 조건으로 평가한다.

nfly 저장소의 가상환경에서 실행한다 (06-1에서 nfly를 RL_tutorial 옆 폴더에 받았다고 가정).

  uv run --project ../nfly ch07/evaluate_nfly.py --model fly --checkpoint ../nfly/runs/fly-cp-s0.pt
  uv run --project ../nfly ch07/evaluate_nfly.py --model mlp --checkpoint ../nfly/runs/mlp-cp-s0.pt
  uv run --project ../nfly ch07/evaluate_nfly.py --model fly --untrained     # 학습 전 초파리 뇌
"""
import argparse
import time

import numpy as np
import torch
from nfly import FlyAgent
from nfly.cli import calibrate_on
from nfly.connectome import load_malecns, select_subset
from nfly.rl.simple.reference import MLPReference
from nfly.suite import get_suite, play_episode


# 가벼운 체크포인트(-slim.pt)에는 커넥톰에서 다시 만들 수 있는 배선 정보가 빠져 있다
REBUILT = {"brain.pre", "brain.post", "brain.sign", "brain.w0"}


def load_agent_state(agent, path):
    payload = torch.load(path, map_location=next(agent.parameters()).device, weights_only=False)
    missing, unexpected = agent.load_state_dict(payload.pop("agent"), strict=False)
    if unexpected or set(missing) - REBUILT:
        raise RuntimeError(f"체크포인트가 이 에이전트와 맞지 않습니다: missing={missing}, unexpected={unexpected}")
    return payload


def build(args, env):
    if args.model == "mlp":
        return MLPReference(env.observation_space, env.action_space, 64)
    conn = select_subset(load_malecns(args.data, min_syn=3), args.subset)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space)
    calibrate_on(agent, get_suite("classic").make("cartpole", seed=args.seed))
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fly", "mlp"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--untrained", action="store_true", help="체크포인트 없이 학습 전 상태로 평가")
    parser.add_argument("--data", default="data", help="MaleCNS feather 파일 폴더 (05-2에서 받은 RL_tutorial/data)")
    parser.add_argument("--subset", default="visual_small")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="학습 때와 같은 값 (readout 보정에 쓰임)")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10,
                        help="평가 전에 관측 정규화 통계를 쌓기 위해 먼저 플레이할 판 수 (점수에 넣지 않음)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    env = get_suite("classic").make("cartpole", seed=args.eval_seed)   # NormalizeObservation 포함
    agent = build(args, env).to(args.device)
    if not args.untrained:
        extra = load_agent_state(agent, args.checkpoint)
        rets = extra.get("returns", [])
        if rets:
            print(f"학습 기록: {len(rets)}판, 마지막 20판 평균 {np.mean(rets[-20:]):.1f}")
    agent.eval()

    # nfly의 classic 환경은 관측을 실시간 평균·분산으로 정규화한다(NormalizeObservation).
    # 이 통계는 체크포인트에 저장되지 않으므로, 먼저 몇 판을 플레이해 통계를 쌓은 뒤 고정한다.
    for ep in range(args.warmup):
        play_episode(agent, env, seed=args.eval_seed - 100 + ep, max_steps=500, device=args.device)
    env.set_wrapper_attr("update_running_mean", False)

    for greedy in (True, False):
        start = time.time()
        returns = [play_episode(agent, env, seed=args.eval_seed + ep, max_steps=500, greedy=greedy,
                                device=args.device).ret for ep in range(args.episodes)]
        returns = np.array(returns)
        mode = "deterministic" if greedy else "stochastic"
        print(f"{args.model} {mode}: {args.episodes}판 평균 {returns.mean():.1f} ± {returns.std():.1f} "
              f"(최소 {returns.min():.0f}, 최대 {returns.max():.0f}, 500점 {int((returns >= 500).sum())}판) "
              f"{time.time() - start:.0f}초")
    env.close()


if __name__ == "__main__":
    main()
