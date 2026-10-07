"""03-2 실습: 가장 단순한 강화학습 게임 Grid World를 직접 만들고 Random Agent로 플레이한다.

  uv run ch03/grid_world.py               # 한 판을 화면에 그리며 실행
  uv run ch03/grid_world.py --episodes 1000   # 1,000판 실행하고 성공률 기록
"""
import argparse
import random

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # 위, 아래, 왼쪽, 오른쪽
ACTION_NAMES = {0: "위", 1: "아래", 2: "왼쪽", 3: "오른쪽"}


class GridWorld:
    """5x5 격자. S에서 출발해 G에 도착하면 성공, X(함정)에 빠지면 실패."""

    def __init__(self, size=5, max_steps=30):
        self.size = size
        self.max_steps = max_steps
        self.goal = (size - 1, size - 1)
        self.traps = {(1, 3), (3, 1), (2, 2)}

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self.pos  # State: 현재 위치

    def step(self, action):
        dr, dc = ACTIONS[action]
        r = min(max(self.pos[0] + dr, 0), self.size - 1)
        c = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (r, c)
        self.steps += 1

        if self.pos == self.goal:
            return self.pos, 1.0, True, "성공"       # 목표 도착: +1
        if self.pos in self.traps:
            return self.pos, -1.0, True, "함정"      # 함정: -1
        if self.steps >= self.max_steps:
            return self.pos, 0.0, True, "시간초과"   # 제한 시간 초과
        return self.pos, -0.01, False, ""            # 한 걸음마다 작은 벌점

    def render(self):
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.pos:
                    row.append("A")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.traps:
                    row.append("X")
                elif (r, c) == (0, 0):
                    row.append("S")
                else:
                    row.append(".")
            print(" ".join(row))
        print()


def random_agent(state):
    return random.choice(list(ACTIONS))


def play_episode(env, agent, render=False):
    state = env.reset()
    total_reward = 0.0
    if render:
        env.render()
    while True:
        action = agent(state)
        state, reward, done, result = env.step(action)
        total_reward += reward
        if render:
            print(f"행동: {ACTION_NAMES[action]}, 보상: {reward:+.2f}")
            env.render()
        if done:
            return result, total_reward, env.steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    env = GridWorld()

    if args.episodes == 1:
        result, total, steps = play_episode(env, random_agent, render=True)
        print(f"결과: {result}, 걸음 수: {steps}, 총 보상: {total:+.2f}")
        return

    counts = {"성공": 0, "함정": 0, "시간초과": 0}
    rewards = []
    for _ in range(args.episodes):
        result, total, _ = play_episode(env, random_agent)
        counts[result] += 1
        rewards.append(total)
    print(f"Random Agent {args.episodes}판 결과")
    for name, n in counts.items():
        print(f"  {name:<6} {n:>5}판 ({n / args.episodes:6.1%})")
    print(f"  평균 총 보상 {sum(rewards) / len(rewards):+.3f}")


if __name__ == "__main__":
    main()
