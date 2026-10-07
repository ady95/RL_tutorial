"""05-4 실습: 커넥톰을 가중치 행렬(희소 행렬)로 바꾼다.

  uv run ch05/to_matrix.py
"""
import numpy as np
import scipy.sparse as sp

import malecns


def main():
    neurons = malecns.load_neurons()
    edges = malecns.load_edges(min_syn=3, neuron_ids=neurons.index)

    # 1. 뉴런 ID(bodyId) → 행렬 번호(0 ~ N-1)
    index = {body: i for i, body in enumerate(neurons.index)}
    pre = edges["body_pre"].map(index).to_numpy()
    post = edges["body_post"].map(index).to_numpy()
    syn = edges["weight"].to_numpy().astype(np.float32)
    n = len(neurons)

    # 2. 부호: 보내는 쪽(pre) 뉴런의 신경전달물질로 정한다 (데일의 법칙)
    sign = neurons["consensus_nt"].map(malecns.NT_SIGN).to_numpy().astype(np.float32)

    # 3. 크기: 받는 쪽(post) 뉴런이 받는 전체 시냅스 중 이 연결의 비율
    total_in = np.bincount(post, weights=syn, minlength=n)
    value = sign[pre] * syn / total_in[post]

    # W[post, pre]: 열 = 보내는 뉴런, 행 = 받는 뉴런
    W = sp.csr_matrix((value, (post, pre)), shape=(n, n))

    print(f"뉴런 {n:,}개, 연결 {W.nnz:,}개")
    print(f"흥분성 연결 비율: {(W.data > 0).mean():.1%}")
    print(f"채워진 칸의 비율(밀도): {W.nnz / n**2:.5%}")
    dense_gb = n * n * 4 / 1e9
    sparse_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1e6
    print(f"메모리: 일반 행렬이라면 {dense_gb:,.0f} GB, 희소 행렬은 {sparse_mb:,.0f} MB")

    in_deg = np.diff(W.indptr)
    out_deg = np.bincount(W.indices, minlength=n)
    print(f"뉴런 하나가 입력을 받는 상대 수: 중앙값 {np.median(in_deg):.0f}, 최대 {in_deg.max():,}")
    print(f"뉴런 하나가 출력을 보내는 상대 수: 중앙값 {np.median(out_deg):.0f}, 최대 {out_deg.max():,}")

    # 4. 되먹임(recurrent) 확인: 서로 연결된 뉴런 쌍
    mutual = W.multiply(W.T)
    print(f"서로 주고받는 뉴런 쌍: {mutual.nnz // 2:,}쌍")

    # 5. 신호 한 단계 전파해 보기: 거대 섬유(DNp01)에 입력을 주는 뉴런을 모두 켜고 W를 곱한다
    gf = index[neurons.index[neurons["type"] == "DNp01"][0]]
    row = W[gf]
    print(f"\nDNp01의 입력: 흥분성 {(row.data > 0).sum()}개, 억제성 {(row.data < 0).sum()}개")
    print(f"  흥분성 입력이 차지하는 시냅스 비율 {row.data[row.data > 0].sum():.3f}, "
          f"억제성 {-row.data[row.data < 0].sum():.3f}")

    h = np.zeros(n, dtype=np.float32)
    h[row.indices] = 1.0                  # DNp01에 입력을 주는 뉴런을 모두 1로 켠다
    h_next = W @ h                        # 행렬 곱 한 번 = 신호가 시냅스를 한 번 건넘
    print(f"  입력 뉴런을 모두 켠 뒤 한 단계: DNp01이 받는 값 {h_next[gf]:+.3f}")
    print(f"  같은 단계에서 0이 아닌 값을 받은 뉴런 {np.count_nonzero(h_next):,}개")


if __name__ == "__main__":
    main()
