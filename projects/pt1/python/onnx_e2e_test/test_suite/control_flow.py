# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX Loop tests."""

import numpy as np
import torch
import onnx
import onnx.helper
import onnx.numpy_helper
from onnx import TensorProto

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

_SHAPE = [2, 3]
# With x in [1, 2) the sum starts in [6, 12), so `y *= 2 while sum(y) < 50`
# stops after 2 to 4 iterations, well before the trip count of 10.
_LIMIT = 50.0


def _const(name, array):
    return onnx.helper.make_node(
        "Constant",
        [],
        [name],
        value=onnx.numpy_helper.from_array(np.array(array), name=name),
    )


def _double_while_below_limit_body():
    """Body ``y_out = y * 2``, ``cond_out = sum(y_out) < _LIMIT``."""
    nodes = [
        _const("two", np.float32(2.0)),
        onnx.helper.make_node("Mul", ["y", "two"], ["y_out"]),
        onnx.helper.make_node("ReduceSum", ["y_out"], ["sum"], keepdims=0),
        _const("limit", np.float32(_LIMIT)),
        onnx.helper.make_node("Less", ["sum", "limit"], ["cond_out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("iter", TensorProto.INT64, []),
        onnx.helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
        onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, _SHAPE),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
        onnx.helper.make_tensor_value_info("y_out", TensorProto.FLOAT, _SHAPE),
    ]
    return onnx.helper.make_graph(nodes, "body", inputs, outputs)


# The initial condition is a constant true while the body computes the
# condition. The loop must stop when sum(y) reaches _LIMIT, before n = 10.
@register_onnx_test
class OnnxLoop_constTrueInitComputedCond(OnnxTestCase):
    @annotate_inputs(
        [
            ("n", torch.int64, [], (10, 11)),
            ("x", torch.float32, _SHAPE, (1.0, 2.0)),
        ]
    )
    def graph(self):
        nodes = [
            _const("cond", np.bool_(True)),
            onnx.helper.make_node(
                "Loop", ["n", "cond", "x"], ["y"], body=_double_while_below_limit_body()
            ),
        ]
        return nodes, ["y"]
