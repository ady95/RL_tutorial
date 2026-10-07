"""02-1 실습 환경 점검: 책에서 쓰는 패키지가 모두 설치되었는지 확인한다."""
import importlib
import platform
import sys

PACKAGES = ["torch", "gymnasium", "stable_baselines3", "pandas", "networkx", "matplotlib", "numpy"]


def main():
    print(f"Python {platform.python_version()} ({platform.system()})")
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  {name:<18} {module.__version__}")
        except ImportError:
            print(f"  {name:<18} 설치되지 않음")
            ok = False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  {'device':<18} {device}")

    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    print(f"  {'CartPole-v1':<18} observation {obs.shape} OK")
    env.close()

    print("환경 점검 통과" if ok else "설치되지 않은 패키지가 있습니다")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
