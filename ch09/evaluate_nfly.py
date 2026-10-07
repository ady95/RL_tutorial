"""09-5 실습: nfly로 학습한 FlyForage 에이전트(초파리 뇌 또는 MLP)를 평가한다.

  uv run --project ../nfly ch09/evaluate_nfly.py --model fly --checkpoint runs/nfly/forage-fly-s0.pt
  uv run --project ../nfly ch09/evaluate_nfly.py --model fly --checkpoint runs/nfly/forage-fly-s0.pt --env FlyForageWide-v0
  uv run --project ../nfly ch09/evaluate_nfly.py --model fly --checkpoint runs/nfly/forage-fly-s0.pt --noise 0.2
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
from nfly import FlyAgent
from nfly.cli import calibrate_on
from nfly.connectome import load_malecns, select_subset
from nfly.rl.simple.reference import MLPReference
from nfly.suite import GameSuite, play_episode

import fly_forage  # noqa: F401

REBUILT = {"brain.pre", "brain.post", "brain.sign", "brain.w0"}   # -slim.pt에서 빠진 배선 정보


def make_env(env_id, noise, seed, normalize):
    # nfly의 gym 묶음과 같은 포장(관측 정규화 + 에피소드 통계)을 하되, 잡음 옵션을 넘긴다
    return GameSuite.finish(gym.make(env_id, obs_noise=noise), seed, normalize_obs=normalize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fly", "mlp"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default="FlyForage-v0")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--data", default="data")
    parser.add_argument("--subset", default="visual_small")
    parser.add_argument("--seed", type=int, default=0, help="학습 때와 같은 값")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10, help="관측 정규화 통계를 쌓는 판 수 (07-3 참조)")
    parser.add_argument("--no-normalize", action="store_true", help="학습 때 --no-normalize를 썼다면 똑같이 준다")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    normalize = not args.no_normalize
    env = make_env(args.env, args.noise, 1000, normalize)
    if args.model == "mlp":
        agent = MLPReference(env.observation_space, env.action_space, 64)
    else:
        conn = select_subset(load_malecns(args.data, min_syn=3), args.subset)
        agent = FlyAgent.build(conn, env.observation_space, env.action_space).to(args.device)
        calibrate_on(agent, make_env("FlyForage-v0", 0.0, args.seed, normalize))
    agent = agent.to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = agent.load_state_dict(payload["agent"], strict=False)
    assert not unexpected and set(missing) <= REBUILT, (missing, unexpected)
    rets = payload.get("returns", [])
    if rets:
        print(f"학습 기록: {len(rets)}판, 마지막 20판 평균 {np.mean(rets[-20:]):.2f}")
    agent.eval()

    if normalize:   # 정규화를 쓸 때만: 통계를 먼저 쌓고 고정한다 (07-3의 함정)
        for ep in range(args.warmup):
            play_episode(agent, env, seed=900 + ep, max_steps=300, device=args.device)
        env.set_wrapper_attr("update_running_mean", False)

    rets, foods, lengths, crashes = [], [], [], 0
    for ep in range(args.episodes):
        res = play_episode(agent, env, seed=1000 + ep, max_steps=300, greedy=True, device=args.device)
        u = env.unwrapped
        rets.append(res.ret)
        foods.append(u.food_eaten)
        lengths.append(u.steps)
        crashes += int(u.steps < u.max_steps)
    rets = np.array(rets)
    print(f"{args.model} @ {args.env} noise={args.noise}: 평균 보상 {rets.mean():.2f} ± {rets.std():.2f} | "
          f"먹이 {np.mean(foods):.1f}개 | 버틴 스텝 {np.mean(lengths):.0f} | 끝까지 생존 {args.episodes - crashes}/{args.episodes}판")


if __name__ == "__main__":
    main()
