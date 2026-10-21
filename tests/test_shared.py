"""Compiled-format guards and weight deduplication without a GPU."""

import json

import msgpack
import pytest
from embeddinggemma2_rocm.shared import externalize_program, shape_bytes


def test_strided_storage_extent():
    assert shape_bytes(dict(type="float_type", lens=[2, 3], strides=[4, 1])) == 28
    assert shape_bytes(dict(type="int64_type", lens=[], strides=[])) == 8
    assert shape_bytes(dict(type="float_type", lens=[0, 3], strides=[3, 1])) == 0
    with pytest.raises(ValueError):
        shape_bytes(dict(sub_shapes=[{}]))


def test_externalized_weights_deduplicated_and_source_preserved(tmp_path):
    shape = dict(type="float_type", lens=[1], strides=[1])
    node = dict(
        name="hip::hip_copy_literal",
        output="@1",
        inputs=[],
        shape=shape,
        operator=dict(literal=dict(shape=shape, data=[b"\0\0\0\0"])),
    )
    raw = msgpack.packb(
        dict(version=7, modules=dict(main=dict(nodes=[node]))), use_bin_type=True
    )
    source = tmp_path / "source.mxr"
    source.write_bytes(raw)
    store = {}
    for name in ("a", "b"):
        target = tmp_path / f"{name}.mxr"
        assert externalize_program(source, target, tmp_path, store) == 0
        converted = msgpack.unpackb(target.read_bytes(), raw=False)
        assert converted["modules"]["main"]["nodes"][0]["name"] == "@param"
        assert (
            json.loads((tmp_path / f"{name}.mxr.memory.json").read_text())[
                "workspace_bytes"
            ]
            == 0
        )
    assert len(store) == 1
    assert len(list((tmp_path / "weights").iterdir())) == 1
    assert source.read_bytes() == raw


def test_unsupported_format_rejected(tmp_path):
    source = tmp_path / "source.mxr"
    source.write_bytes(msgpack.packb(dict(version=8)))
    with pytest.raises(ValueError, match="version 7"):
        externalize_program(source, tmp_path / "output.mxr", tmp_path, {})
