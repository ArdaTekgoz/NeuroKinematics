import json
from pathlib import Path

import numpy as np
import pytest

from neurokinematics.data.factory import (canonical_array_hash, read_shard, sha256_file,
    verify_dataset, write_deterministic_npz)


def test_deterministic_file_and_content_hash(tmp_path, valid_arrays, schema):
    order = [x["name"] for x in schema["fields"]]
    a, b = tmp_path/"a.npz", tmp_path/"b.npz"
    write_deterministic_npz(a, valid_arrays, order); write_deterministic_npz(b, valid_arrays, order)
    assert sha256_file(a) == sha256_file(b)
    assert canonical_array_hash(read_shard(a, order), order) == canonical_array_hash(read_shard(b, order), order)


def test_row_order_changes_content_hash(valid_arrays, schema):
    order = [x["name"] for x in schema["fields"]]
    reversed_rows = {k:v[::-1].copy() for k,v in valid_arrays.items()}
    assert canonical_array_hash(valid_arrays, order) != canonical_array_hash(reversed_rows, order)


def test_truncated_or_byte_corrupted_shard_rejected(tmp_path, valid_arrays, schema):
    order = [x["name"] for x in schema["fields"]]
    path = tmp_path/"shard.npz"; write_deterministic_npz(path, valid_arrays, order)
    path.write_bytes(path.read_bytes()[:50])
    with pytest.raises(ValueError): read_shard(path, order)
