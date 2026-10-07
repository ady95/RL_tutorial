"""05-2 실습: MaleCNS의 뉴런 표와 연결 표를 열어 본다.

  uv run ch05/explore_tables.py
"""
import time

import pandas as pd

import malecns


def main():
    pd.set_option("display.width", 120)

    ann = pd.read_feather(malecns.DATA / "body-annotations.feather")
    print(f"[body-annotations] {len(ann):,}행 x {ann.shape[1]}열")
    print("  superclass가 있는 몸체(뉴런):", f"{ann['superclass'].notna().sum():,}")

    neurons = malecns.load_neurons()
    print(f"\n[뉴런 표] {len(neurons):,}개, 세포 유형(type) {neurons['type'].nunique():,}종")
    print(neurons.loc[[10001], ["type", "instance", "superclass", "somaSide", "consensus_nt"]].to_string())

    print("\n[superclass별 뉴런 수, 상위 12개]")
    print(neurons["superclass"].value_counts().head(12).to_string())

    print("\n[신경전달물질(consensus_nt)별 뉴런 수]")
    print(neurons["consensus_nt"].value_counts().to_string())

    start = time.time()
    edges = malecns.load_edges(min_syn=3, neuron_ids=neurons.index)
    print(f"\n[연결 표] 뉴런 사이, 시냅스 3개 이상: {len(edges):,}개 ({time.time() - start:.1f}초)")
    print(edges.sort_values("weight", ascending=False).head(5).to_string(index=False))
    print("\n시냅스 수 분포:")
    print(edges["weight"].describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())

    top = edges.nlargest(1, "weight").iloc[0]
    pre, post = neurons.loc[top.body_pre], neurons.loc[top.body_post]
    print(f"\n가장 강한 연결: {pre['type']}({pre['superclass']}) → {post['type']}({post['superclass']}), "
          f"시냅스 {top.weight:,}개")


if __name__ == "__main__":
    main()
