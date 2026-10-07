"""04-4: 학습된 PPO 에이전트의 신경망 구조와 파라미터 수를 들여다본다.

  uv run ch04/inspect_policy.py runs/ppo_cartpole/model.zip
"""
import argparse

import torch
from stable_baselines3 import PPO


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="runs/ppo_cartpole/model.zip")
    args = parser.parse_args()

    model = PPO.load(args.model)
    policy = model.policy
    print(policy)

    print("\n부분별 파라미터 수")
    parts = {
        "policy_net (행동 쪽 은닉층)": policy.mlp_extractor.policy_net,
        "value_net 은닉층 (가치 쪽)": policy.mlp_extractor.value_net,
        "action_net (행동 출력층)": policy.action_net,
        "value_net (가치 출력층)": policy.value_net,
    }
    for name, module in parts.items():
        print(f"  {name:<28} {count(module):>6,}")
    print(f"  {'합계':<28} {count(policy):>6,}")

    obs = torch.tensor([[0.0, 0.0, 0.05, 0.0]])  # 막대가 오른쪽으로 살짝 기운 상태
    with torch.no_grad():
        dist = policy.get_distribution(obs)
        value = policy.predict_values(obs)
    probs = dist.distribution.probs[0].tolist()
    print(f"\n막대가 오른쪽으로 기운 상태 → 왼쪽 {probs[0]:.3f}, 오른쪽 {probs[1]:.3f}, 가치 {value.item():.1f}")


if __name__ == "__main__":
    main()
