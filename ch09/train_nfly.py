"""09-4 실습: nfly의 PPO 학습기로 FlyForage를 학습시킨다 (초파리 뇌 또는 같은 학습기의 MLP).

nfly 저장소의 가상환경에서 실행한다. RL_tutorial 폴더에서:
  uv run --project ../nfly ch09/train_nfly.py --model mlp --seed 0 --updates 400 --device cpu
  uv run --project ../nfly ch09/train_nfly.py --model fly --seed 0 --updates 400 --device cuda
"""
import argparse

import gymnasium as gym
import torch
from nfly import FlyAgent
from nfly.cli import calibrate_on
from nfly.connectome import load_malecns, select_subset
from nfly.rl import PPOConfig, train_ppo
from nfly.rl.simple.reference import MLPReference
from nfly.suite import GameSuite, get_suite

import fly_forage  # noqa: F401  (FlyForage-v0 등록: gym.make로 부를 수 있게 됨)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fly", "mlp"], required=True)
    parser.add_argument("--env", default="FlyForage-v0")
    parser.add_argument("--data", default="data", help="MaleCNS feather 파일 폴더 (05-2)")
    parser.add_argument("--subset", default="visual_small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=400, help="업데이트 1번 = 8개 환경 x rollout 스텝")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--entropy", type=float, default=0.0, help="엔트로피 보너스 (nfly 기본 0)")
    parser.add_argument("--gamma", type=float, default=0.98, help="할인율 (nfly 기본 0.98)")
    parser.add_argument("--lam", type=float, default=0.8, help="GAE 람다 (nfly 기본 0.8)")
    parser.add_argument("--rollout", type=int, default=32, help="환경 하나당 한 번에 모으는 스텝 수 (nfly 기본 32)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="관측 정규화(NormalizeObservation)를 끈다. FlyForage의 관측은 이미 -1~1 범위")
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    suite = get_suite("gym")                       # 등록된 Gymnasium id라면 무엇이든 받는 nfly 게임 묶음
    if args.no_normalize:
        def make_env(seed):
            return lambda: GameSuite.finish(gym.make(args.env), seed, normalize_obs=False)
        venv = gym.vector.SyncVectorEnv([make_env(args.seed + i) for i in range(8)],
                                        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        venv = suite.make_vector(args.env, 8, seed=args.seed)
    if args.model == "mlp":
        agent = MLPReference(venv.single_observation_space, venv.single_action_space, 64)
    else:
        conn = select_subset(load_malecns(args.data, min_syn=3), args.subset)
        agent = FlyAgent.build(conn, venv.single_observation_space, venv.single_action_space).to(args.device)
        calib_env = (GameSuite.finish(gym.make(args.env), args.seed, normalize_obs=False) if args.no_normalize
                     else suite.make(args.env, seed=args.seed))
        calibrate_on(agent, calib_env)
    agent = agent.to(args.device)
    print(agent.summary())

    out = args.out or f"runs/nfly/forage-{args.model}-s{args.seed}.pt"
    cfg = PPOConfig(updates=args.updates, out=out, entropy=args.entropy, gamma=args.gamma, lam=args.lam,
                    rollout=args.rollout)
    train_ppo(agent, venv, cfg, device=args.device, seed=args.seed)
    print(f"체크포인트 저장: {out}")


if __name__ == "__main__":
    main()
