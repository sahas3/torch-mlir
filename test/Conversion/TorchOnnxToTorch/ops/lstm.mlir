// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch -cse | FileCheck %s



// CHECK-LABEL:   func.func @test_lstm_basic(
// CHECK-SAME:                               %[[X:.*]]: !torch.vtensor<[15,2,4],f32>,
// CHECK-SAME:                               %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[R:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[B:.*]]: !torch.vtensor<[1,24],f32>)
// CHECK:           %[[LOOP_RESULT:.*]]:3 = torch.prim.Loop %[[MAX_TRIPS:.*]], %[[ENTER_LOOP:.*]], init(%[[Y:.*]], %[[INITIAL_H:.*]], %[[INITIAL_C:.*]]) {
// CHECK:           ^bb0(%[[LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV:.*]]: !torch.vtensor<[15,2,3],f32>, %[[H_PREV:.*]]: !torch.vtensor<[2,3],f32>, %[[C_PREV:.*]]: !torch.vtensor<[2,3],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK-DAG:           }
// CHECK:         }

func.func @test_lstm_basic(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL:   func.func @test_lstm_bidirectional_with_initial_bias(
// CHECK-SAME:                                               %[[X:.*]]: !torch.vtensor<[32,32,192],f32>,
// CHECK-SAME:                                               %[[W:.*]]: !torch.vtensor<[2,192,192],f32>,
// CHECK-SAME:                                               %[[R:.*]]: !torch.vtensor<[2,192,48],f32>,
// CHECK-SAME:                                               %[[B:.*]]: !torch.vtensor<[2,384],f32>)
// CHECK:           %[[FORWARD_LOOP_RES:.*]]:3 = torch.prim.Loop %[[MAX_TRIP_FWD:.*]], %[[LOOP_COND_FWD:.*]], init(%[[Y_FWD:.*]], %[[INITIAL_H_FWD:.*]], %[[INITIAL_C_FWD:.*]]) {
// CHECK:           ^bb0(%[[FORWARD_LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV_FWD:.*]]: !torch.vtensor<[32,32,48],f32>, %[[H_PREV_FWD:.*]]: !torch.vtensor<[32,48],f32>, %[[C_PREV_FWD:.*]]: !torch.vtensor<[32,48],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK:           }
// CHECK:           torch.aten.flip
// CHECK:           %[[REVERSE_LOOP_RES:.*]]:3 = torch.prim.Loop %[[MAX_TRIPS_REV:.*]], %[[LOOP_COND_REV:.*]], init(%[[Y_REV:.*]], %[[INITIAL_H_REV:.*]], %[[INITIAL_C_REV:.*]]) {
// CHECK:           ^bb0(%[[REVERSE_LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV_REV:.*]]: !torch.vtensor<[32,32,48],f32>, %[[H_PREV_REV:.*]]: !torch.vtensor<[32,48],f32>, %[[C_PREV_REV:.*]]: !torch.vtensor<[32,48],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK:           }
// CHECK:           torch.aten.flip
// CHECK:           return %[[Y:.*]], %[[Y_H:.*]], %[[Y_C:.*]] : !torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>
// CHECK:         }

func.func @test_lstm_bidirectional_with_initial_bias(%arg0: !torch.vtensor<[32,32,192],f32>, %arg1: !torch.vtensor<[2,192,192],f32>, %arg2: !torch.vtensor<[2,192,48],f32>, %arg3: !torch.vtensor<[2,384],f32>) -> (!torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.direction = "bidirectional", torch.onnx.hidden_size = 48 : si64, torch.onnx.layout = 0 : si64} : (!torch.vtensor<[32,32,192],f32>, !torch.vtensor<[2,192,192],f32>, !torch.vtensor<[2,192,48],f32>, !torch.vtensor<[2,384],f32>) -> (!torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>
}

// -----

// CHECK-LABEL:   func.func @test_lstm_batchwise_two_outputs(
// CHECK-SAME:                                               %[[X_LAYOUT_1:.*]]: !torch.vtensor<[3,1,2],f32>,
// CHECK-SAME:                                               %[[W:.*]]: !torch.vtensor<[1,28,2],f32>,
// CHECK-SAME:                                               %[[R:.*]]: !torch.vtensor<[1,28,7],f32>)
// CHECK:       torch.aten.transpose.int
// CHECK:           %[[LOOP_RES:.*]]:3 = torch.prim.Loop %[[MAX_TRIP:.*]], %[[LOOP_COND_FWD:.*]], init(%[[Y:.*]], %[[INITIAL_H:.*]], %[[INITIAL_C:.*]]) {
// CHECK:           ^bb0(%[[LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV:.*]]: !torch.vtensor<[1,3,7],f32>, %[[H_PREV:.*]]: !torch.vtensor<[3,7],f32>, %[[C_PREV:.*]]: !torch.vtensor<[3,7],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK:           }
// CHECK-DAG:           torch.aten.transpose.int
// CHECK-DAG:           torch.aten.transpose.int
// CHECK-DAG:           torch.aten.transpose.int
// CHECK-DAG:           torch.aten.transpose.int
// CHECK:           return %[[Y:.*]], %[[Y_H:.*]] : !torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>
// CHECK:         }

func.func @test_lstm_batchwise_two_outputs(%arg0: !torch.vtensor<[3,1,2],f32>, %arg1: !torch.vtensor<[1,28,2],f32>, %arg2: !torch.vtensor<[1,28,7],f32>) -> (!torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 7 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1,28,2],f32>, !torch.vtensor<[1,28,7],f32>) -> (!torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>
}

// -----

// CHECK-LABEL:   func.func @test_lstm_dynamic(
// CHECK-SAME:                               %[[X:.*]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:                               %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[R:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[B:.*]]: !torch.vtensor<[1,24],f32>)
// CHECK:           torch.runtime.assert %[[EQ:.*]], "The input_size of W must equal X."
// CHECK:           %[[LOOP_RESULT:.*]]:3 = torch.prim.Loop %[[MAX_TRIPS:.*]], %[[ENTER_LOOP:.*]], init(%[[Y:.*]], %[[INITIAL_H:.*]], %[[INITIAL_C:.*]]) {
// CHECK:           ^bb0(%[[LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV:.*]]: !torch.vtensor<[?,?,3],f32>, %[[H_PREV:.*]]: !torch.vtensor<[?,3],f32>, %[[C_PREV:.*]]: !torch.vtensor<[?,3],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK-DAG:           }
// CHECK:         }

 func.func @test_lstm_dynamic(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// When initial_h / initial_c are provided with a dynamic hidden dim (e.g. from
// an onnx.Tile broadcasting the initial state to a dynamic batch dim), the
// expander must recover the static hidden_size from the attribute so the loop
// body carries the static hidden dim instead of propagating an unknown size
// that leaves an unresolvable materialization against the static result type.

// CHECK-LABEL:   func.func @test_lstm_dynamic_initial_state(
// CHECK-SAME:                               %[[X:.*]]: !torch.vtensor<[?,?,4],f32>,
// CHECK-SAME:                               %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[R:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[B:.*]]: !torch.vtensor<[1,24],f32>,
// CHECK-SAME:                               %[[INITIAL_H:.*]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:                               %[[INITIAL_C:.*]]: !torch.vtensor<[?,?,?],f32>)
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[INITIAL_H]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %[[INITIAL_C]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_FWD:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[C0_FWD]]) {
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,?,3],f32>, %{{.*}}: !torch.vtensor<[?,3],f32>, %{{.*}}: !torch.vtensor<[?,3],f32>):
// CHECK:           }

func.func @test_lstm_dynamic_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[?,?,?],f32>, %arg5: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// When initial_h / initial_c already carry a static hidden dim, no refinement
// cast is emitted: the dim is only refined when it is dynamic. This also guards
// against emitting a verifier-invalid cast for a statically-mismatched dim.

// CHECK-LABEL:   func.func @test_lstm_static_initial_state(
// CHECK-SAME:                               %[[INITIAL_H:.*]]: !torch.vtensor<[1,?,3],f32>,
// CHECK-SAME:                               %[[INITIAL_C:.*]]: !torch.vtensor<[1,?,3],f32>)
// CHECK-NOT:       torch.tensor_static_info_cast %[[INITIAL_H]]
// CHECK-NOT:       torch.tensor_static_info_cast %[[INITIAL_C]]

func.func @test_lstm_static_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[1,?,3],f32>, %arg5: !torch.vtensor<[1,?,3],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// layout=1 with a dynamic initial state: the layout transpose runs before the
// refinement, so the hidden dim (still the trailing dim after the transpose) is
// pinned to the static hidden_size on the transposed value.

// CHECK-LABEL:   func.func @test_lstm_layout1_dynamic_initial_state(
// CHECK:           %[[TH:.*]] = torch.aten.transpose.int %arg4, %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[TH]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[TC:.*]] = torch.aten.transpose.int %arg5, %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %[[TC]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_FWD:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[C0_FWD]])

func.func @test_lstm_layout1_dynamic_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[?,?,?],f32>, %arg5: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,1,3],f32>, !torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64, torch.onnx.layout = 1 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>)-> (!torch.vtensor<[?,?,1,3],f32>, !torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,?,1,3],f32>, !torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>
}

// -----

// bidirectional with a dynamic initial state: the refinement runs once on the
// combined [num_directions, batch, hidden] operand before the forward/reverse
// split, pinning the hidden dim to the static hidden_size.

// CHECK-LABEL:   func.func @test_lstm_bidirectional_dynamic_initial_state(
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %arg4 : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %arg5 : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_FWD:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[H0_REV:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_REV:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[C0_FWD]])
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_REV]], %[[C0_REV]])

func.func @test_lstm_bidirectional_dynamic_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[2,12,4],f32>, %arg2: !torch.vtensor<[2,12,3],f32>, %arg3: !torch.vtensor<[2,24],f32>, %arg4: !torch.vtensor<[?,?,?],f32>, %arg5: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,2,?,3],f32>, !torch.vtensor<[2,?,3],f32>, !torch.vtensor<[2,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64, torch.onnx.direction = "bidirectional" }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[2,12,4],f32>, !torch.vtensor<[2,12,3],f32>, !torch.vtensor<[2,24],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>)-> (!torch.vtensor<[?,2,?,3],f32>, !torch.vtensor<[2,?,3],f32>, !torch.vtensor<[2,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,2,?,3],f32>, !torch.vtensor<[2,?,3],f32>, !torch.vtensor<[2,?,3],f32>
}

// -----

// A statically-batched initial_h/initial_c (batch=1, as emitted by common ONNX
// exporters) combined with a dynamic X batch must not bake the loop-carried
// hidden/cell state to batch 1: the state batch tracks X's batch. The provided
// states are cast to the dynamic-batch state type before the loop, and each
// gate matmul against the dynamic-batch Xt stays dynamic ([?,3], not [1,3]).

// CHECK-LABEL:   func.func @test_lstm_static_batch_dynamic_x(
// CHECK:           %[[H0_DIR:.*]] = torch.aten.select.int %arg4, %{{.*}}, %{{.*}} : !torch.vtensor<[1,1,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,3],f32>
// CHECK:           %[[C0_DIR:.*]] = torch.aten.select.int %arg5, %{{.*}}, %{{.*}} : !torch.vtensor<[1,1,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,3],f32>
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[H0_DIR]] : !torch.vtensor<[1,3],f32> to !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %[[C0_DIR]] : !torch.vtensor<[1,3],f32> to !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0]], %[[C0]])
// CHECK:           torch.aten.linear %{{.*}} : !torch.vtensor<[?,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[?,3],f32>
// CHECK-NOT:       -> !torch.vtensor<[1,3],f32>

func.func @test_lstm_static_batch_dynamic_x(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[1,1,3],f32>, %arg5: !torch.vtensor<[1,1,3],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[1,1,3],f32>, !torch.vtensor<[1,1,3],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// A declared Y more static than the computed one (a static Y for a dynamic X):
// Y is cast to the declared type; Y_h and Y_c already match and are not cast.

// CHECK-LABEL:   func.func @test_lstm_static_result_dynamic_x(
// CHECK:           %[[YH:.*]] = torch.aten.unsqueeze %{{.*}}#1, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[YC:.*]] = torch.aten.unsqueeze %{{.*}}#2, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y:.*]] = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[?,1,2,3],f32> to !torch.vtensor<[5,1,2,3],f32>
// CHECK:           return %[[Y]], %[[YH]], %[[YC]] : !torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>

func.func @test_lstm_static_result_dynamic_x(%arg0: !torch.vtensor<[?,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[?,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Dynamic input size in both X and W: the two sizes are checked at run time.

// CHECK-LABEL:   func.func @test_lstm_dynamic_input_size_dynamic_w(
// CHECK-DAG:       %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[IN:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[W_IN:.*]] = torch.aten.size.int %arg1, %{{.*}} : {{.*}} -> !torch.int
// CHECK:           %[[EQ:.*]] = torch.aten.eq.int %[[IN]], %[[W_IN]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[EQ]], "The input_size of W must equal X."
// CHECK:           torch.prim.Loop

func.func @test_lstm_dynamic_input_size_dynamic_w(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[1,12,?],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1,12,?],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// Static input size in X with a dynamic W: W's input size is checked at run
// time against X's.

// CHECK-LABEL:   func.func @test_lstm_static_input_size_dynamic_w(
// CHECK-DAG:       %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[W_IN:.*]] = torch.aten.size.int %arg1, %{{.*}} : {{.*}} -> !torch.int
// CHECK:           %[[EQ:.*]] = torch.aten.eq.int %[[C5]], %[[W_IN]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[EQ]], "The input_size of W must equal X."
// CHECK:           torch.prim.Loop

func.func @test_lstm_static_input_size_dynamic_w(%arg0: !torch.vtensor<[3,2,5],f32>, %arg1: !torch.vtensor<[1,12,?],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,5],f32>, !torch.vtensor<[1,12,?],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Declared results more static than the computed ones (static results for a
// dynamic X batch): Y, Y_h and Y_c are each cast to their declared type.

// CHECK-LABEL:   func.func @test_lstm_static_results_dynamic_batch(
// CHECK:           %[[LOOP:.*]]:3 = torch.prim.Loop
// CHECK:           %[[YH_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#1, %{{.*}} : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.vtensor<[1,?,3],f32>
// CHECK:           %[[YC_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#2, %{{.*}} : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.vtensor<[1,?,3],f32>
// CHECK:           %[[Y_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#0, %{{.*}} : !torch.vtensor<[5,?,3],f32>, !torch.int -> !torch.vtensor<[5,1,?,3],f32>
// CHECK:           %[[Y:.*]] = torch.tensor_static_info_cast %[[Y_U]] : !torch.vtensor<[5,1,?,3],f32> to !torch.vtensor<[5,1,2,3],f32>
// CHECK:           %[[YH:.*]] = torch.tensor_static_info_cast %[[YH_U]] : !torch.vtensor<[1,?,3],f32> to !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[YC:.*]] = torch.tensor_static_info_cast %[[YC_U]] : !torch.vtensor<[1,?,3],f32> to !torch.vtensor<[1,2,3],f32>
// CHECK:           return %[[Y]], %[[YH]], %[[YC]] :

func.func @test_lstm_static_results_dynamic_batch(%arg0: !torch.vtensor<[5,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[5,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[5,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// A reverse LSTM runs the layer with the direction-0 parameters on X flipped
// along the seq dim, then flips Y back. Y_h and Y_c are the loop's final
// states, the states after time step 0.

// CHECK-LABEL:   func.func @test_lstm_reverse(
// CHECK-SAME:                                 %[[X:.*]]: !torch.vtensor<[15,2,4],f32>,
// CHECK-SAME:                                 %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK:           torch.aten.select.int %[[W]], %[[C0]], %[[C0]] : !torch.vtensor<[1,12,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,4],f32>
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[C0]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[X_REV:.*]] = torch.aten.flip %[[X]], %[[DIMS]] : !torch.vtensor<[15,2,4],f32>, !torch.list<int> -> !torch.vtensor<[15,2,4],f32>
// CHECK:           %[[LOOP:.*]]:3 = torch.prim.Loop
// CHECK:             torch.aten.select.int %[[X_REV]], %{{.*}}, %{{.*}} : !torch.vtensor<[15,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK:           %[[Y_H:.*]] = torch.aten.unsqueeze %[[LOOP]]#1, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y_C:.*]] = torch.aten.unsqueeze %[[LOOP]]#2, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#0, %{{.*}} : !torch.vtensor<[15,2,3],f32>, !torch.int -> !torch.vtensor<[15,1,2,3],f32>
// CHECK:           %[[Y:.*]] = torch.aten.flip %[[Y_U]], %[[DIMS]] : !torch.vtensor<[15,1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[15,1,2,3],f32>
// CHECK:           return %[[Y]], %[[Y_H]], %[[Y_C]]

func.func @test_lstm_reverse(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.direction = "reverse", torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// With layout 1, X is transposed to [seq, batch, input] before the reverse
// direction flips it along the seq dim, and Y, Y_h and Y_c are transposed back
// to [batch, ...] after Y is flipped back.

// CHECK-LABEL:   func.func @test_lstm_layout1_reverse(
// CHECK-SAME:                                         %[[X:.*]]: !torch.vtensor<[2,3,4],f32>,
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK:           %[[X_T:.*]] = torch.aten.transpose.int %[[X]], %{{.*}}, %{{.*}} : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,2,4],f32>
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[C0]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[X_REV:.*]] = torch.aten.flip %[[X_T]], %[[DIMS]] : !torch.vtensor<[3,2,4],f32>, !torch.list<int> -> !torch.vtensor<[3,2,4],f32>
// CHECK:           %[[LOOP:.*]]:3 = torch.prim.Loop
// CHECK:             torch.aten.select.int %[[X_REV]], %{{.*}}, %{{.*}} : !torch.vtensor<[3,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK:           %[[Y_H:.*]] = torch.aten.unsqueeze %[[LOOP]]#1, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y_C:.*]] = torch.aten.unsqueeze %[[LOOP]]#2, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#0, %{{.*}} : !torch.vtensor<[3,2,3],f32>, !torch.int -> !torch.vtensor<[3,1,2,3],f32>
// CHECK:           %[[Y:.*]] = torch.aten.flip %[[Y_U]], %[[DIMS]] : !torch.vtensor<[3,1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[3,1,2,3],f32>
// CHECK:           %[[Y_T1:.*]] = torch.aten.transpose.int %[[Y]], %{{.*}}, %{{.*}} : !torch.vtensor<[3,1,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,2,1,3],f32>
// CHECK:           %[[Y_T:.*]] = torch.aten.transpose.int %[[Y_T1]], %{{.*}}, %{{.*}} : !torch.vtensor<[3,2,1,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3,1,3],f32>
// CHECK:           %[[Y_H_T:.*]] = torch.aten.transpose.int %[[Y_H]], %{{.*}}, %{{.*}} : !torch.vtensor<[1,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,1,3],f32>
// CHECK:           %[[Y_C_T:.*]] = torch.aten.transpose.int %[[Y_C]], %{{.*}}, %{{.*}} : !torch.vtensor<[1,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,1,3],f32>
// CHECK:           return %[[Y_T]], %[[Y_H_T]], %[[Y_C_T]]
func.func @test_lstm_layout1_reverse(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>) -> (!torch.vtensor<[2,3,1,3],f32>, !torch.vtensor<[2,1,3],f32>, !torch.vtensor<[2,1,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2) {torch.onnx.direction = "reverse", torch.onnx.hidden_size = 3 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>) -> (!torch.vtensor<[2,3,1,3],f32>, !torch.vtensor<[2,1,3],f32>, !torch.vtensor<[2,1,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[2,3,1,3],f32>, !torch.vtensor<[2,1,3],f32>, !torch.vtensor<[2,1,3],f32>
}

// -----

// Dynamic dim 1 of W: only a static size is checked against 4 * hidden_size.
// CHECK-LABEL: func.func @test_lstm_dynamic_w_hidden
// CHECK-NOT: torch.operator
// CHECK: torch.prim.Loop
func.func @test_lstm_dynamic_w_hidden(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,?,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,?,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// A bidirectional LSTM runs direction 0 forward and direction 1 in reverse,
// each with its own slice of W, R, B, initial_h and initial_c, and stacks Y,
// Y_h and Y_c forward first.

// CHECK-LABEL:   func.func @test_lstm_bidirectional(
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[B0:.*]] = torch.aten.select.int %arg3, %[[C0]], %[[C0]] : !torch.vtensor<[2,24],f32>, !torch.int, !torch.int -> !torch.vtensor<[24],f32>
// CHECK:           %[[WB0:.*]] = torch.aten.slice.Tensor %[[B0]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[24],f32>, {{.*}} -> !torch.vtensor<[12],f32>
// CHECK:           %[[RB0:.*]] = torch.aten.slice.Tensor %[[B0]],
// CHECK:           %[[BI0:.*]] = torch.aten.slice.Tensor %[[WB0]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[12],f32>, {{.*}} -> !torch.vtensor<[3],f32>
// CHECK:           %[[BO0:.*]] = torch.aten.slice.Tensor %[[WB0]],
// CHECK:           %[[BF0:.*]] = torch.aten.slice.Tensor %[[WB0]],
// CHECK:           %[[BC0:.*]] = torch.aten.slice.Tensor %[[WB0]],
// CHECK:           %[[RBI0:.*]] = torch.aten.slice.Tensor %[[RB0]],
// CHECK:           %[[RBO0:.*]] = torch.aten.slice.Tensor %[[RB0]],
// CHECK:           %[[RBF0:.*]] = torch.aten.slice.Tensor %[[RB0]],
// CHECK:           %[[RBC0:.*]] = torch.aten.slice.Tensor %[[RB0]],
// CHECK:           %[[W0:.*]] = torch.aten.select.int %arg1, %[[C0]], %[[C0]] : !torch.vtensor<[2,12,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,4],f32>
// CHECK:           %[[WI0:.*]] = torch.aten.slice.Tensor %[[W0]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[12,4],f32>, {{.*}} -> !torch.vtensor<[3,4],f32>
// CHECK:           %[[WO0:.*]] = torch.aten.slice.Tensor %[[W0]],
// CHECK:           %[[WF0:.*]] = torch.aten.slice.Tensor %[[W0]],
// CHECK:           %[[WC0:.*]] = torch.aten.slice.Tensor %[[W0]],
// CHECK:           %[[R0:.*]] = torch.aten.select.int %arg2, %[[C0]], %[[C0]] : !torch.vtensor<[2,12,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,3],f32>
// CHECK:           %[[RI0:.*]] = torch.aten.slice.Tensor %[[R0]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[12,3],f32>, {{.*}} -> !torch.vtensor<[3,3],f32>
// CHECK:           %[[RO0:.*]] = torch.aten.slice.Tensor %[[R0]],
// CHECK:           %[[RF0:.*]] = torch.aten.slice.Tensor %[[R0]],
// CHECK:           %[[RC0:.*]] = torch.aten.slice.Tensor %[[R0]],
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %arg4, %[[C0]], %[[C0]] : !torch.vtensor<[2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[CELL0_FWD:.*]] = torch.aten.select.int %arg5, %[[C0]], %[[C0]] : !torch.vtensor<[2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[B1:.*]] = torch.aten.select.int %arg3, %[[C0]], %[[C1]] : !torch.vtensor<[2,24],f32>, !torch.int, !torch.int -> !torch.vtensor<[24],f32>
// CHECK:           %[[WB1:.*]] = torch.aten.slice.Tensor %[[B1]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[24],f32>, {{.*}} -> !torch.vtensor<[12],f32>
// CHECK:           %[[RB1:.*]] = torch.aten.slice.Tensor %[[B1]],
// CHECK:           %[[BI1:.*]] = torch.aten.slice.Tensor %[[WB1]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[12],f32>, {{.*}} -> !torch.vtensor<[3],f32>
// CHECK:           %[[BO1:.*]] = torch.aten.slice.Tensor %[[WB1]],
// CHECK:           %[[BF1:.*]] = torch.aten.slice.Tensor %[[WB1]],
// CHECK:           %[[BC1:.*]] = torch.aten.slice.Tensor %[[WB1]],
// CHECK:           %[[RBI1:.*]] = torch.aten.slice.Tensor %[[RB1]],
// CHECK:           %[[RBO1:.*]] = torch.aten.slice.Tensor %[[RB1]],
// CHECK:           %[[RBF1:.*]] = torch.aten.slice.Tensor %[[RB1]],
// CHECK:           %[[RBC1:.*]] = torch.aten.slice.Tensor %[[RB1]],
// CHECK:           %[[W1:.*]] = torch.aten.select.int %arg1, %[[C0]], %[[C1]] : !torch.vtensor<[2,12,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,4],f32>
// CHECK:           %[[WI1:.*]] = torch.aten.slice.Tensor %[[W1]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[12,4],f32>, {{.*}} -> !torch.vtensor<[3,4],f32>
// CHECK:           %[[WO1:.*]] = torch.aten.slice.Tensor %[[W1]],
// CHECK:           %[[WF1:.*]] = torch.aten.slice.Tensor %[[W1]],
// CHECK:           %[[WC1:.*]] = torch.aten.slice.Tensor %[[W1]],
// CHECK:           %[[R1:.*]] = torch.aten.select.int %arg2, %[[C0]], %[[C1]] : !torch.vtensor<[2,12,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,3],f32>
// CHECK:           %[[RI1:.*]] = torch.aten.slice.Tensor %[[R1]], %[[C0]], %[[C0]], %{{.*}}, %[[C1]] : !torch.vtensor<[12,3],f32>, {{.*}} -> !torch.vtensor<[3,3],f32>
// CHECK:           %[[RO1:.*]] = torch.aten.slice.Tensor %[[R1]],
// CHECK:           %[[RF1:.*]] = torch.aten.slice.Tensor %[[R1]],
// CHECK:           %[[RC1:.*]] = torch.aten.slice.Tensor %[[R1]],
// CHECK:           %[[H0_REV:.*]] = torch.aten.select.int %arg4, %[[C0]], %[[C1]] : !torch.vtensor<[2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[CELL0_REV:.*]] = torch.aten.select.int %arg5, %[[C0]], %[[C1]] : !torch.vtensor<[2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[FWD:.*]]:3 = torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[CELL0_FWD]])
// CHECK:             torch.aten.select.int %arg0, %{{.*}}, %{{.*}} : !torch.vtensor<[3,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK:             torch.aten.linear %{{.*}}, %[[WI0]], %[[BI0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RI0]], %[[RBI0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[WO0]], %[[BO0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RO0]], %[[RBO0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[WF0]], %[[BF0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RF0]], %[[RBF0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[WC0]], %[[BC0]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RC0]], %[[RBC0]]
// CHECK:           %[[YH_FWD:.*]] = torch.aten.unsqueeze %[[FWD]]#1, %[[C0]] : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[YC_FWD:.*]] = torch.aten.unsqueeze %[[FWD]]#2, %[[C0]] : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y_FWD:.*]] = torch.aten.unsqueeze %[[FWD]]#0, %[[C1]] : !torch.vtensor<[3,2,3],f32>, !torch.int -> !torch.vtensor<[3,1,2,3],f32>
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[C0]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[X_REV:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[3,2,4],f32>, !torch.list<int> -> !torch.vtensor<[3,2,4],f32>
// CHECK:           %[[REV:.*]]:3 = torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_REV]], %[[CELL0_REV]])
// CHECK:             torch.aten.select.int %[[X_REV]], %{{.*}}, %{{.*}} : !torch.vtensor<[3,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK:             torch.aten.linear %{{.*}}, %[[WI1]], %[[BI1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RI1]], %[[RBI1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[WO1]], %[[BO1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RO1]], %[[RBO1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[WF1]], %[[BF1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RF1]], %[[RBF1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[WC1]], %[[BC1]]
// CHECK:             torch.aten.linear %{{.*}}, %[[RC1]], %[[RBC1]]
// CHECK:           %[[YH_REV:.*]] = torch.aten.unsqueeze %[[REV]]#1, %[[C0]] : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[YC_REV:.*]] = torch.aten.unsqueeze %[[REV]]#2, %[[C0]] : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[Y_REV_U:.*]] = torch.aten.unsqueeze %[[REV]]#0, %[[C1]] : !torch.vtensor<[3,2,3],f32>, !torch.int -> !torch.vtensor<[3,1,2,3],f32>
// CHECK:           %[[Y_REV:.*]] = torch.aten.flip %[[Y_REV_U]], %[[DIMS]] : !torch.vtensor<[3,1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[3,1,2,3],f32>
// CHECK:           %[[Y_LIST:.*]] = torch.prim.ListConstruct %[[Y_FWD]], %[[Y_REV]] : (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[3,1,2,3],f32>) -> !torch.list<vtensor<[3,1,2,3],f32>>
// CHECK:           %[[Y:.*]] = torch.aten.cat %[[Y_LIST]], %[[C1]] : !torch.list<vtensor<[3,1,2,3],f32>>, !torch.int -> !torch.vtensor<[3,2,2,3],f32>
// CHECK:           %[[YH_LIST:.*]] = torch.prim.ListConstruct %[[YH_FWD]], %[[YH_REV]] : (!torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) -> !torch.list<vtensor<[1,2,3],f32>>
// CHECK:           %[[Y_H:.*]] = torch.aten.cat %[[YH_LIST]], %[[C0]] : !torch.list<vtensor<[1,2,3],f32>>, !torch.int -> !torch.vtensor<[2,2,3],f32>
// CHECK:           %[[YC_LIST:.*]] = torch.prim.ListConstruct %[[YC_FWD]], %[[YC_REV]] : (!torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) -> !torch.list<vtensor<[1,2,3],f32>>
// CHECK:           %[[Y_C:.*]] = torch.aten.cat %[[YC_LIST]], %[[C0]] : !torch.list<vtensor<[1,2,3],f32>>, !torch.int -> !torch.vtensor<[2,2,3],f32>
// CHECK:           return %[[Y]], %[[Y_H]], %[[Y_C]] : !torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>

func.func @test_lstm_bidirectional(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[2,12,4],f32>, %arg2: !torch.vtensor<[2,12,3],f32>, %arg3: !torch.vtensor<[2,24],f32>, %arg4: !torch.vtensor<[2,2,3],f32>, %arg5: !torch.vtensor<[2,2,3],f32>) -> (!torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) {torch.onnx.direction = "bidirectional", torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[2,12,4],f32>, !torch.vtensor<[2,12,3],f32>, !torch.vtensor<[2,24],f32>, !torch.none, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>) -> (!torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>
}
