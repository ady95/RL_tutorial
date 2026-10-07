"""06-4 실습: CartPole용 초파리 Brain Agent를 조립하고 부위별로 해부한다.

nfly 저장소의 가상환경에서 실행한다 (06-1에서 nfly를 RL_tutorial 옆 폴더에 받았다고 가정).

  uv run --project ../nfly ch06/inspect_agent.py                       # visual_small 부분 회로
  uv run --project ../nfly ch06/inspect_agent.py --subset all          # 중추신경계 전체
"""
import argparse
import time
from collections import Counter

import torch
from nfly import FlyAgent
from nfly.cli import calibrate_on
from nfly.connectome import load_malecns, select_subset
from nfly.suite import get_suite


def count(params):
    return sum(p.numel() for p in params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="MaleCNS feather 파일 폴더 (05-2에서 받은 RL_tutorial/data)")
    parser.add_argument("--subset", default="visual_small")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(0)

    # 1. 커넥톰 불러오기 → 부분 회로 고르기
    conn = select_subset(load_malecns(args.data, min_syn=3), args.subset)
    env = get_suite("classic").make("cartpole", seed=0)

    # 2. 게임의 관측·행동 공간에 맞춰 에이전트 조립 (Encoder · Brain · Decoder 자동 선택)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space).to(args.device)
    calibrate_on(agent, get_suite("classic").make("cartpole", seed=0))
    print(agent.summary())

    # 3. 입구와 출구가 어떤 뉴런인가
    sc = conn.neurons["super_class"].to_numpy()
    enc_idx = agent.encoder.idx.cpu().numpy()
    dec_idx = agent.decoder.idx.cpu().numpy()
    print(f"\n입력 뉴런 {len(enc_idx)}개의 분류:", dict(Counter(sc[enc_idx]).most_common()))
    print(f"출력(readout) 뉴런 {len(dec_idx)}개의 분류:", dict(Counter(sc[dec_idx]).most_common()))

    # 4. 학습되는 파라미터는 어디에 있나
    print("\n부위별 학습 파라미터")
    for name, module in [("encoder", agent.encoder), ("brain", agent.brain),
                         ("decoder", agent.decoder), ("value (critic)", agent.value)]:
        print(f"  {name:<16} {count(module.parameters()):>10,}")
        for pname, p in module.named_parameters():
            print(f"      {pname:<28} {tuple(p.shape)}")
    print(f"  {'input_gain':<16} {agent.input_gain.numel():>10,}")
    print(f"  {'합계':<16} {count(agent.parameters()):>10,}")

    print("\n고정된 값(buffer) 중 큰 것")
    for bname, b in agent.named_buffers():
        if b.numel() > 1000:
            print(f"  {bname:<28} {tuple(b.shape)}")

    # 5. 한 스텝 계산 시간 (게임 한 스텝 = 신경망 4단계)
    obs, _ = env.reset(seed=0)
    h = agent.initial_state(1)
    x = torch.as_tensor(obs, device=args.device).unsqueeze(0)
    agent.act(x, h)
    start = time.time()
    for _ in range(20):
        _, h = agent.act(x, h)
    print(f"\n게임 1스텝 계산 시간({args.device}): {(time.time() - start) / 20 * 1000:.0f} ms")


if __name__ == "__main__":
    main()
