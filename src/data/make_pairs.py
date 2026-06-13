import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR       = Path("data/processed")
PASSAGES_FILE  = DATA_DIR / "msmarco_small_passages.json"
QUERIES_FILE   = DATA_DIR / "msmarco_small_queries.json"
QRELS_FILE     = DATA_DIR / "msmarco_small_qrels.json"

OUT_TRAIN = DATA_DIR / "train_pairs.jsonl"
OUT_VAL   = DATA_DIR / "val_pairs.jsonl"
OUT_TEST  = DATA_DIR / "test_pairs.jsonl"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):,} pairs → {path}")


def build_passage_lookup(passages):
    return {p["id"]: p["text"] for p in passages}


def make_pairs(queries, qrels, passage_lookup):
    pairs = []
    for query in queries:
        qid   = query["id"]
        qtext = query["text"].strip()

        if qid not in qrels:
            continue

        best_pid, _ = max(qrels[qid], key=lambda x: x[1])
        if best_pid not in passage_lookup:
            continue

        ptext = passage_lookup[best_pid].strip()
        if len(qtext.split()) < 3 or len(ptext.split()) < 10:
            continue

        pairs.append({"query": qtext, "positive": ptext})
    return pairs


def split(pairs):
    random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def main():
    print("Loading data …")
    passages       = load_json(PASSAGES_FILE)
    queries        = load_json(QUERIES_FILE)
    qrels          = load_json(QRELS_FILE)
    passage_lookup = build_passage_lookup(passages)

    print("Building pairs …")
    pairs = make_pairs(queries, qrels, passage_lookup)
    print(f"  total pairs: {len(pairs):,}")

    train, val, test = split(pairs)
    print(f"  train/val/test: {len(train)}/{len(val)}/{len(test)}")

    write_jsonl(train, OUT_TRAIN)
    write_jsonl(val,   OUT_VAL)
    write_jsonl(test,  OUT_TEST)
    print("Done")


if __name__ == "__main__":
    main()
