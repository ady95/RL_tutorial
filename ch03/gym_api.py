"""03-3: Gymnasium 환경의 구조를 하나씩 출력해 본다.

  uv run ch03/gym_api.py
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    print("환경 이름        :", env.spec.id)
    print("최대 스텝 수     :", env.spec.max_episode_steps)
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)
    print("행동 개수        :", env.action_space.n)

    obs, info = env.reset(seed=42)
    print("\nreset() → obs    :", obs)
    print("reset() → info   :", info)

    obs, reward, terminated, truncated, info = env.step(1)
    print("\nstep(1) → obs    :", obs)
    print("         reward  :", reward)
    print("     terminated  :", terminated)
    print("      truncated  :", truncated)

    env.action_space.seed(0)
    print("\nobservation_space.contains(obs):", env.observation_space.contains(obs))
    print("action_space.sample() 5번      :", [int(env.action_space.sample()) for _ in range(5)])

    # 500스텝을 채우면 terminated가 아니라 truncated로 끝난다는 것을 확인
    obs, info = env.reset(seed=42)
    steps = 0
    while True:
        # 막대가 기운 쪽으로 수레를 미는 간단한 규칙 (각도 + 각속도)
        action = 1 if obs[2] + 0.5 * obs[3] > 0 else 0
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    print(f"\n규칙 에이전트: {steps}스텝, terminated={terminated}, truncated={truncated}")
    env.close()


if __name__ == "__main__":
    main()
