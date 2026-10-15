from __future__ import annotations

import torch


def assert_nested_equal(testcase, left, right):
    if isinstance(left, torch.Tensor):
        testcase.assertTrue(torch.equal(left, right), msg=f"{left!r} != {right!r}")
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(testcase, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            assert_nested_equal(testcase, left_item, right_item)
    else:
        testcase.assertEqual(left, right)

