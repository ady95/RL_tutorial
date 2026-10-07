"""10-1~10-3 실습: 배선을 바꾼 초파리 뇌가 CartPole에 필요한 정보를 출구(하행 뉴런)까지 전달하는지 잰다.

강화학습은 시드에 따라 결과가 크게 흔들려(07-3) 조건 간 차이를 보기 어렵다. 그래서 nfly 개발자가 쓴
"충분성 검사"를 쓴다: 학습하지 않은 뇌의 출구 활동으로 규칙 에이전트(03-3)의 행동을 지도학습으로
따라 하게 하고, 그 선형 헤드로 실제로 플레이해 본다. 몇 분이면 끝나고 결과가 안정적이다.

nfly 가상환경에서 실행한다:
  uv run --project ../nfly ch10/wiring_probe.py --variant real
  uv run --project ../nfly ch10/wiring_probe.py --variant ablate --fraction 0.3
  uv run --project ../nfly ch10/wiring_probe.py --variant shuffle
  uv run --project ../nfly ch10/wiring_probe.py --variant random
  uv run --project ../nfly ch10/wiring_probe.py --variant real --subset visual
  uv run --project ../nfly ch10/wiring_probe.py --variant real --subset eye_to_dn   # 눈과 하행 뉴런만
"""
import argparse
import time

import numpy as np
import torch
from nfly import FlyAgent
from nfly.cli import calibrate_on
from nfly.connectome import load_malecns, select_subset
from nfly.connectome.base import build_connectome
from nfly.suite import get_suite

PROTECTED = ("ol_sensory", "descending_neuron")     # 입구와 출구는 지우지 않는다


def load_subset(data, name):
    """nfly의 부분 회로(all, brain, visual, visual_small)에 더해, 중간 회로를 모두 뺀 eye_to_dn을 만든다."""
    conn = load_malecns(data, min_syn=3)
    if name == "eye_to_dn":
        keep = np.flatnonzero(conn.neurons["super_class"].isin(PROTECTED).to_numpy())
        return conn.subset(torch.as_tensor(keep))
    return select_subset(conn, name)


def neuron_sign(conn):
    """뉴런마다의 부호 (데일의 법칙: 보내는 뉴런의 모든 연결은 같은 부호)."""
    s = torch.ones(conn.n_neurons)
    s[conn.pre] = conn.sign
    return s


def make_variant(conn, variant, fraction, rng):
    if variant == "real":
        return conn
    if variant == "ablate":
        sc = conn.neurons["super_class"].to_numpy()
        middle = np.flatnonzero(~np.isin(sc, PROTECTED))
        drop = rng.choice(middle, size=int(len(middle) * fraction), replace=False)
        keep = np.setdiff1d(np.arange(conn.n_neurons), drop)
        return conn.subset(torch.as_tensor(keep))
    sign = neuron_sign(conn)
    if variant == "shuffle":
        # 연결마다 "보내는 뉴런"만 뒤섞는다: 뉴런별 입력·출력 연결 수는 그대로, 누가 누구와 이어지는지만 바뀜
        pre = conn.pre[torch.as_tensor(rng.permutation(conn.n_edges))]
        post = conn.post
    elif variant == "random":
        # 같은 수의 연결을 완전히 무작위로 다시 긋는다
        pre = torch.as_tensor(rng.integers(0, conn.n_neurons, conn.n_edges))
        post = torch.as_tensor(rng.integers(0, conn.n_neurons, conn.n_edges))
    else:
        raise ValueError(variant)
    syn = conn.syn_count[torch.as_tensor(rng.permutation(conn.n_edges))]
    return build_connectome(conn.neurons, pre, post, syn, sign[pre])


def teacher(state):
    """03-3의 규칙 에이전트: 막대 각도 + 0.5 x 각속도의 부호로 민다."""
    return int(state[2] + 0.5 * state[3] > 0)


@torch.no_grad()
def rollout(agent, env, steps, explore, rng, device, head=None):
    feats, labels, returns = [], [], []
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    h, total = agent.initial_state(1), 0.0
    for _ in range(steps):
        f, h = agent.step(torch.as_tensor(obs, device=device).unsqueeze(0).float(), h)
        a_teacher = teacher(env.unwrapped.state)
        if head is None:
            feats.append(f[0].cpu()); labels.append(a_teacher)
            a = int(rng.integers(2)) if rng.random() < explore else a_teacher
        else:
            a = int((f @ head[0] + head[1]).argmax())
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            returns.append(total); total = 0.0
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            h = agent.initial_state(1)
    return (torch.stack(feats), torch.as_tensor(labels)) if head is None else returns


def fit_linear_head(x, y, epochs=300):
    """특징 → 행동 2개의 선형 분류기 (로지스틱 회귀)를 경사하강법으로 맞춘다."""
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    w = torch.zeros(x.shape[1], 2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, y)
        loss.backward(); opt.step()
    acc = float(((x @ w + b).argmax(1) == y).float().mean())
    return w.detach(), b.detach(), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["real", "ablate", "shuffle", "random"], default="real")
    parser.add_argument("--fraction", type=float, default=0.1, help="ablate: 지울 중간 뉴런 비율")
    parser.add_argument("--subset", default="visual_small")
    parser.add_argument("--data", default="data")
    parser.add_argument("--steps", type=int, default=3000, help="규칙 에이전트를 따라 하며 모을 스텝 수")
    parser.add_argument("--play-steps", type=int, default=3000, help="학습한 헤드로 플레이할 총 스텝 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.time()

    conn = load_subset(args.data, args.subset)
    conn = make_variant(conn, args.variant, args.fraction, rng)
    env = get_suite("classic").make("cartpole", seed=args.seed)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space).to(args.device)
    r2 = agent.calibrate_on_env(get_suite("classic").make("cartpole", seed=args.seed + 1))
    print(agent.summary())

    x, y = rollout(agent, env, args.steps, explore=0.3, rng=rng, device=args.device)
    w, b, acc = fit_linear_head(x, y)
    mean, std = x.mean(0).to(args.device), x.std(0).to(args.device) + 1e-6
    # 정규화를 헤드에 합쳐 둔다: (f - mean) / std @ w + b
    head = ((w.to(args.device) / std.unsqueeze(1)), b.to(args.device) - (mean / std) @ w.to(args.device))
    returns = rollout(agent, env, args.play_steps, explore=0.0, rng=rng, device=args.device, head=head)

    label = args.variant + (f" {args.fraction:.0%}" if args.variant == "ablate" else "") + f" ({args.subset})"
    print(f"\n[{label}] 출구 보정 R2 {r2:.3f} | 규칙 따라 하기 정확도 {acc:.1%} | "
          f"헤드로 플레이 {len(returns)}판 평균 {np.mean(returns) if returns else float('nan'):.1f}"
          f"{' (끝까지 버팀)' if not returns else ''} | {time.time() - start:.0f}초")


if __name__ == "__main__":
    main()
