"""5장 공통: MaleCNS 파일을 읽는 함수 모음. 다른 예제가 import malecns 로 가져다 쓴다."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

DATA = Path("data")

# 신경전달물질 → 시냅스 부호 (nfly와 같은 규칙: 흥분성 +1, 억제성 -1, 불분명은 +1)
NT_SIGN = {
    "acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
    "gaba": -1, "glutamate": -1, "histamine": -1, "unclear": 1,
}


def load_neurons():
    """뉴런 표: superclass가 있는 몸체(=뉴런)만 남기고 신경전달물질을 붙인다."""
    ann = pd.read_feather(DATA / "body-annotations.feather")
    ann = ann[ann["superclass"].notna()]
    nt = pd.read_feather(DATA / "body-neurotransmitters.feather", columns=["body", "consensus_nt"])
    neurons = ann.merge(nt, left_on="bodyId", right_on="body", how="left").drop(columns="body")
    neurons["consensus_nt"] = neurons["consensus_nt"].fillna("unclear")
    return neurons.set_index("bodyId")


def load_edges(min_syn=3, neuron_ids=None):
    """연결 표를 조각(record batch) 단위로 읽으며 시냅스 수가 min_syn 이상인 연결만 남긴다.

    1억 5천만 행을 한꺼번에 읽으면 메모리를 수 GB 쓰므로, 조각마다 걸러서 모은다.
    """
    reader = ipc.open_file(DATA / "connectome-weights.feather")
    ids = pa.array(list(neuron_ids)) if neuron_ids is not None else None
    parts = []
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        mask = pc.greater_equal(batch["weight"], min_syn)
        if ids is not None:
            mask = pc.and_(mask, pc.and_(pc.is_in(batch["body_pre"], ids), pc.is_in(batch["body_post"], ids)))
        parts.append(batch.filter(mask))
    return pa.Table.from_batches(parts).to_pandas()
