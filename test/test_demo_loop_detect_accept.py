#!/usr/bin/env python3
"""验收脚本的回归测试。"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import demo_loop_detect_accept as accept

_SQLITE_C = "/home/gumble/dev/3rd/sqlite/test/c"
_SQLITE_JSON = "/home/gumble/dev/3rd/sqlite/test/json"


def test_default_cycle_lens_cover_required_range():
    lens = accept._default_cycle_lens(400)
    assert lens == sorted(set(lens))
    assert lens[0] >= 100
    assert lens[-1] == 400
    assert 133 in lens
    assert 266 in lens
    assert 387 in lens
    assert all(2 <= value <= 400 for value in lens)


def test_cycle_lens_override():
    assert accept._parse_cycle_lens("133,266,387", 400) == [133, 266, 387]
    assert accept._parse_cycle_lens(" 5, 5, 400 ", 400) == [5, 400]
    assert accept._parse_cycle_lens(None, 400) == accept._default_cycle_lens(400)


def test_docs_smoke_acceptance():
    rc = accept.main([
        "--docs-trials", "1",
        "--max-cycle-rounds", "20",
        "--quiet",
    ])
    assert rc == 0


def test_sqlite_smoke_acceptance():
    rc = accept.main([
        "--sqlite", _SQLITE_C,
        "--sqlite", _SQLITE_JSON,
        "--docs-trials", "1",
        "--sqlite-trials", "1",
        "--max-files", "20",
        "--max-cycle-rounds", "20",
        "--quiet",
    ])
    assert rc == 0
