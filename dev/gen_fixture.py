#!/usr/bin/env python3

import json
import sys
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto, checker, helper, numpy_helper
except ImportError:
    print("Missing Python package: onnx. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Missing Python package: numpy. It is installed with onnx; try: pip install onnx", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
VOCAB = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "foo": 4, "bar": 5}


def embedding_table():
    table = np.zeros((100, 4), dtype=np.float32)
    for i in range(100):
        table[i] = [i, i + 0.5, -i, i * 0.1]
    return table


def checked_model(graph):
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17)],
        producer_name="embeddings-clj-fixtures",
    )
    model.ir_version = 10
    checker.check_model(model)
    return model


def tokenizer_json():
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": VOCAB,
            "unk_token": "[UNK]",
        },
    }


def save_tokenizer(model_dir):
    with (model_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_json(), f, indent=2, sort_keys=True)
        f.write("\n")


def make_token_model(model_dir):
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    output = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["batch", "seq", 4]
    )
    table = numpy_helper.from_array(embedding_table(), name="embedding_table")
    gather = helper.make_node(
        "Gather",
        inputs=["embedding_table", "input_ids"],
        outputs=["last_hidden_state"],
        axis=0,
    )
    graph = helper.make_graph(
        [gather],
        "token-model",
        [input_ids, attention_mask],
        [output],
        [table],
    )
    onnx.save(checked_model(graph), model_dir / "model.onnx")


def make_pooled_model(model_dir):
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    output = helper.make_tensor_value_info(
        "sentence_embedding", TensorProto.FLOAT, ["batch", 4]
    )
    table = numpy_helper.from_array(embedding_table(), name="embedding_table")
    gather = helper.make_node(
        "Gather",
        inputs=["embedding_table", "input_ids"],
        outputs=["token_embeddings"],
        axis=0,
    )
    reduce_mean = helper.make_node(
        "ReduceMean",
        inputs=["token_embeddings"],
        outputs=["sentence_embedding"],
        axes=[1],
        keepdims=0,
    )
    graph = helper.make_graph(
        [gather, reduce_mean],
        "pooled-model",
        [input_ids, attention_mask],
        [output],
        [table],
    )
    onnx.save(checked_model(graph), model_dir / "model.onnx")


def main():
    token_dir = FIXTURES / "token-model"
    pooled_dir = FIXTURES / "pooled-model"
    token_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)

    make_token_model(token_dir)
    make_pooled_model(pooled_dir)
    save_tokenizer(token_dir)
    save_tokenizer(pooled_dir)

    print(f"wrote {token_dir / 'model.onnx'}")
    print(f"wrote {pooled_dir / 'model.onnx'}")


if __name__ == "__main__":
    main()
