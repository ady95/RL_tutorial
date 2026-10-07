"""08-2 실습: MaleCNS에서 버섯체(Mushroom Body) 회로를 찾아 뉴런 수와 연결을 센다.

  uv run ch08/mushroom_body.py

뉴런을 고르는 기준(세포 유형 이름 패턴)은 FlyTris와 같다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch05"))
import malecns  # noqa: E402  (05장의 데이터 읽기 함수)

PATTERNS = {
    "Kenyon Cell (KC)": r"^KC",
    "APL": r"^APL$",
    "MBON": r"^MBON",
    "PAM (보상 도파민)": r"^PAM\d",
    "PPL1 (처벌 도파민)": r"^PPL1\d",
}


def main():
    neurons = malecns.load_neurons()
    groups = {}
    print("[버섯체 회로의 뉴런]")
    for name, pattern in PATTERNS.items():
        mask = neurons["type"].fillna("").str.match(pattern)
        groups[name] = set(neurons.index[mask])
        types = neurons.loc[mask, "type"].nunique()
        nts = neurons.loc[mask, "consensus_nt"].value_counts().head(2).to_dict()
        print(f"  {name:<18} {mask.sum():>6,}개  세포 유형 {types:>3}종  전달물질 {nts}")

    ids = set().union(*groups.values())
    edges = malecns.load_edges(min_syn=3, neuron_ids=ids)
    print(f"\n[버섯체 회로 안의 연결] 시냅스 3개 이상: {len(edges):,}개")

    def between(a, b):
        e = edges[edges["body_pre"].isin(groups[a]) & edges["body_post"].isin(groups[b])]
        return len(e), int(e["weight"].sum())

    for a, b in [("Kenyon Cell (KC)", "MBON"), ("Kenyon Cell (KC)", "APL"), ("APL", "Kenyon Cell (KC)"),
                 ("PAM (보상 도파민)", "MBON"), ("PPL1 (처벌 도파민)", "MBON"),
                 ("PAM (보상 도파민)", "Kenyon Cell (KC)"), ("MBON", "PAM (보상 도파민)")]:
        n, syn = between(a, b)
        print(f"  {a:<18} → {b:<18} 연결 {n:>7,}개, 시냅스 {syn:>9,}개")

    kc = groups["Kenyon Cell (KC)"]
    kc_in = edges[edges["body_post"].isin(kc)]
    kc_out_mbon = edges[edges["body_pre"].isin(kc) & edges["body_post"].isin(groups["MBON"])]
    print(f"\nKenyon Cell 하나가 출력을 보내는 MBON 수: 중앙값 "
          f"{kc_out_mbon.groupby('body_pre').size().median():.0f}")
    print(f"MBON 하나가 입력을 받는 Kenyon Cell 수: 중앙값 "
          f"{kc_out_mbon.groupby('body_post').size().median():.0f}")
    print(f"(참고) 버섯체 회로 안에서 Kenyon Cell로 들어오는 연결: {len(kc_in):,}개")


if __name__ == "__main__":
    main()
