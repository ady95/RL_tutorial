"""05-2 실습: MaleCNS v1.0 커넥톰 파일 3개를 data/ 폴더에 내려받는다 (약 1.2GB, CC-BY 4.0).

  uv run ch05/download_malecns.py

파일 이름은 6장의 nfly가 쓰는 이름과 같게 저장하므로, 6장에서 다시 받을 필요가 없다.
데이터 출처: https://male-cns.janelia.org/  (Berg et al., Cell 2026)
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "body-annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",       # 약 14MB
    "body-neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",       # 약 43MB
    "connectome-weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",   # 약 1.1GB
}


def download(url, dest):
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and dest.stat().st_size == total:
            print(f"  이미 있음: {dest.name} ({total / 1e6:,.0f} MB)")
            return
        tmp = dest.with_suffix(".part")
        done = 0
        with open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
        tmp.replace(dest)
        print()


def main():
    data = Path("data")
    data.mkdir(exist_ok=True)
    for name, remote in FILES.items():
        download(f"{BASE}/{remote}", data / name)
    print("완료:", ", ".join(sorted(p.name for p in data.glob("*.feather"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
