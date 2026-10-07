"""09-2 실습: 먹이를 받고 장애물을 피하는 게임 FlyForage를 Gymnasium 환경으로 만든다.

    맨 위 줄에서 먹이(*)와 장애물(#)이 생겨 한 스텝에 한 칸씩 떨어진다.
    초파리(F)는 맨 아래 줄에서 왼쪽 / 정지 / 오른쪽으로 움직인다.

    . . * . . . .
    . . . . # . .
    . . . . . . .
    . # . . . . .
    . . . . . . .
    . . . . . * .
    . . . . . . .
    . . . F . . .

보상: 먹이를 받으면 +1, 장애물에 부딪히면 -1(게임 종료), 버틴 스텝마다 +0.01

  uv run ch09/fly_forage.py            # 무작위 행동으로 한 판을 화면에 그리기
  uv run ch09/fly_forage.py --check    # Gymnasium 규격 검사
"""
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LEFT, STAY, RIGHT = 0, 1, 2


class FlyForageEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, width=7, height=8, p_food=0.3, p_obstacle=0.3, max_steps=300,
                 obs_noise=0.0, food_reward=1.0, crash_penalty=-1.0, survive_reward=0.01,
                 render_mode=None):
        self.width, self.height = width, height
        self.p_food, self.p_obstacle = p_food, p_obstacle
        self.max_steps = max_steps
        self.obs_noise = obs_noise
        self.food_reward, self.crash_penalty, self.survive_reward = food_reward, crash_penalty, survive_reward
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        # [초파리 위치, 가장 가까운 먹이의 가로 거리·세로 거리, 가장 가까운 장애물의 가로 거리·세로 거리]
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fly = self.width // 2
        self.objects = []          # [행, 열, 종류] 종류: 1 = 먹이, -1 = 장애물
        self.steps = 0
        self.food_eaten = 0
        return self._obs(), self._info()

    def step(self, action):
        self.fly = int(np.clip(self.fly + (action - 1), 0, self.width - 1))   # 0→-1, 1→0, 2→+1

        reward, terminated = self.survive_reward, False
        for obj in self.objects:
            obj[0] += 1                                    # 한 칸 떨어진다
        landed = [o for o in self.objects if o[0] == self.height - 1]
        self.objects = [o for o in self.objects if o[0] < self.height - 1]
        for row, col, kind in landed:
            if col != self.fly:
                continue
            if kind == 1:
                reward += self.food_reward
                self.food_eaten += 1
            else:
                reward += self.crash_penalty
                terminated = True

        self._spawn()
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        return self._obs(), reward, terminated, truncated, self._info()

    def _spawn(self):
        cols = self.np_random.permutation(self.width)
        if self.np_random.random() < self.p_food:
            self.objects.append([0, int(cols[0]), 1])
        if self.np_random.random() < self.p_obstacle:
            self.objects.append([0, int(cols[1]), -1])

    def _nearest(self, kind):
        """종류가 kind인 물체 중 가장 아래(가장 먼저 떨어질) 것의 상대 위치. 없으면 (0, 1)."""
        cands = [o for o in self.objects if o[2] == kind]
        if not cands:
            return 0.0, 1.0
        row, col, _ = max(cands, key=lambda o: (o[0], -abs(o[1] - self.fly)))
        dx = (col - self.fly) / (self.width - 1)
        dy = (self.height - 1 - row) / (self.height - 1)
        return dx, dy

    def _obs(self):
        fx = 2.0 * self.fly / (self.width - 1) - 1.0
        obs = np.array([fx, *self._nearest(1), *self._nearest(-1)], dtype=np.float32)
        if self.obs_noise > 0:
            obs += self.np_random.normal(0, self.obs_noise, size=obs.shape).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _info(self):
        return {"food_eaten": self.food_eaten}

    def render(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for row, col, kind in self.objects:
            grid[row][col] = "*" if kind == 1 else "#"
        grid[self.height - 1][self.fly] = "F"
        return "\n".join(" ".join(r) for r in grid)


# gym.make("FlyForage-v0")로 부를 수 있게 등록한다
gym.register("FlyForage-v0", entry_point=FlyForageEnv)
# 10장의 "새로운 맵": 더 넓고 장애물이 더 많은 판
gym.register("FlyForageWide-v0", entry_point=FlyForageEnv, kwargs={"width": 9, "p_obstacle": 0.45})
# 10-4의 보상 실험: 같은 게임, 다른 보상
gym.register("FlyForageFoodOnly-v0", entry_point=FlyForageEnv, kwargs={"crash_penalty": 0.0, "survive_reward": 0.0})
gym.register("FlyForageSafety-v0", entry_point=FlyForageEnv, kwargs={"food_reward": 0.1, "crash_penalty": -10.0})
gym.register("FlyForageSurvive-v0", entry_point=FlyForageEnv, kwargs={"food_reward": 0.0})


def main():
    if "--check" in sys.argv:
        from gymnasium.utils.env_checker import check_env
        check_env(gym.make("FlyForage-v0").unwrapped)
        print("check_env OK")
        return
    env = gym.make("FlyForage-v0", render_mode="ansi")
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            break
    print(env.unwrapped.render())
    print(f"\n{env.unwrapped.steps}스텝, 먹이 {info['food_eaten']}개, 총 보상 {total:.2f}, "
          f"{'충돌' if terminated else '시간 종료'}")


if __name__ == "__main__":
    main()
