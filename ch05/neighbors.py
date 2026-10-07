"""05-3 실습: 특정 뉴런의 입력(upstream)·출력(downstream) 상대와 연결 경로를 찾는다.

  uv run ch05/neighbors.py                 # 기본: 거대 섬유 뉴런 DNp01
  uv run ch05/neighbors.py --type MDN      # 다른 세포 유형
"""
import argparse

import networkx as nx

import malecns


def describe(neurons, ids):
    rows = neurons.loc[ids, ["type", "superclass", "consensus_nt"]]
    return rows.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="DNp01", help="찾을 세포 유형 (기본: 거대 섬유 Giant Fiber)")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--path-min-syn", type=int, default=10, help="경로 탐색에 쓸 연결의 최소 시냅스 수")
    args = parser.parse_args()

    neurons = malecns.load_neurons()
    edges = malecns.load_edges(min_syn=3, neuron_ids=neurons.index)

    # 1. 세포 유형으로 뉴런 찾기
    targets = neurons.index[neurons["type"] == args.type].tolist()
    print(f"[{args.type}] 뉴런 {len(targets)}개:", targets)
    target = targets[0]
    print(neurons.loc[[target], ["instance", "superclass", "consensus_nt"]].to_string())

    # 2. upstream: 이 뉴런에 입력을 주는 뉴런 (pandas로 충분하다)
    up = edges[edges["body_post"] == target].nlargest(args.top, "weight")
    up = up.merge(describe(neurons, up["body_pre"]), left_on="body_pre", right_on="bodyId")
    print(f"\n입력을 주는 뉴런 {int((edges['body_post'] == target).sum())}개 중 상위 {args.top}개")
    print(up[["body_pre", "type", "superclass", "consensus_nt", "weight"]].to_string(index=False))

    # 3. downstream: 이 뉴런이 출력을 보내는 뉴런
    down = edges[edges["body_pre"] == target].nlargest(args.top, "weight")
    down = down.merge(describe(neurons, down["body_post"]), left_on="body_post", right_on="bodyId")
    print(f"\n출력을 받는 뉴런 {int((edges['body_pre'] == target).sum())}개 중 상위 {args.top}개")
    print(down[["body_post", "type", "superclass", "consensus_nt", "weight"]].to_string(index=False))

    # 4. 연결 경로: 강한 연결만 남긴 그래프에서 감각 뉴런 → target 최단 경로 (NetworkX)
    strong = edges[edges["weight"] >= args.path_min_syn]
    G = nx.from_pandas_edgelist(strong, "body_pre", "body_post", edge_attr="weight", create_using=nx.DiGraph)
    print(f"\n그래프: 시냅스 {args.path_min_syn}개 이상 연결만, 노드 {G.number_of_nodes():,}개, 엣지 {G.number_of_edges():,}개")

    sensory = set(neurons.index[neurons["superclass"].str.contains("sensory")]) & set(G.nodes)
    G_rev = G.reverse(copy=False)
    dist = nx.single_source_shortest_path_length(G_rev, target)   # target까지 몇 단계 만에 닿는가
    reach = {n: d for n, d in dist.items() if n in sensory}
    if not reach:
        print("감각 뉴런에서 닿는 경로가 없습니다")
        return
    print(f"{args.type}에 닿는 감각 뉴런 {len(reach):,}개, 최소 {min(reach.values())}단계")
    hop_counts = {}
    for d in reach.values():
        hop_counts[d] = hop_counts.get(d, 0) + 1
    print("  단계별 감각 뉴런 수:", dict(sorted(hop_counts.items())))

    source = min(reach, key=lambda n: (reach[n], -G[n][next(iter(G[n]))]["weight"]))
    path = nx.shortest_path(G, source, target)
    print("\n가장 짧은 경로 예:")
    for a, b in zip(path, path[1:] + [None]):
        info = neurons.loc[a]
        line = f"  {a} {info['type']} ({info['superclass']}, {info['consensus_nt']})"
        if b is not None:
            line += f"  --{G[a][b]['weight']}개-->"
        print(line)


if __name__ == "__main__":
    main()
