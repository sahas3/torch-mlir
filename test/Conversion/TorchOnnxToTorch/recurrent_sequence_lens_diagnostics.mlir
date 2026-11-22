// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// A bidirectional LSTM with sequence_lens is a match failure: sequence_lens is not
// supported yet.
func.func @test_lstm_bidirectional_sequence_lens(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[2,12,4],f32>, %arg2: !torch.vtensor<[2,12,3],f32>, %arg3: !torch.vtensor<[2,24],f32>, %arg4: !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.direction = "bidirectional", torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[2,12,4],f32>, !torch.vtensor<[2,12,3],f32>, !torch.vtensor<[2,24],f32>, !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>
}

// -----

// A reverse GRU with sequence_lens is a match failure: sequence_lens is not
// supported yet.
func.func @test_gru_reverse_sequence_lens(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,9,4],f32>, %arg2: !torch.vtensor<[1,9,3],f32>, %arg3: !torch.vtensor<[1,18],f32>, %arg4: !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.direction = "reverse", torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,9,4],f32>, !torch.vtensor<[1,9,3],f32>, !torch.vtensor<[1,18],f32>, !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// A reverse RNN with sequence_lens is a match failure: sequence_lens is not
// supported yet.
func.func @test_rnn_reverse_sequence_lens(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>, %arg3: !torch.vtensor<[1,6],f32>, %arg4: !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.direction = "reverse", torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>, !torch.vtensor<[1,6],f32>, !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// A forward LSTM with sequence_lens is a match failure: sequence_lens is not
// supported yet.
func.func @test_lstm_forward_sequence_lens(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// A forward GRU with sequence_lens is a match failure: sequence_lens is not
// supported yet.
func.func @test_gru_forward_sequence_lens(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,9,4],f32>, %arg2: !torch.vtensor<[1,9,3],f32>, %arg3: !torch.vtensor<[1,18],f32>, %arg4: !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,9,4],f32>, !torch.vtensor<[1,9,3],f32>, !torch.vtensor<[1,18],f32>, !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// A forward RNN with sequence_lens is a match failure: sequence_lens is not
// supported yet.
func.func @test_rnn_forward_sequence_lens(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>, %arg3: !torch.vtensor<[1,6],f32>, %arg4: !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>, !torch.vtensor<[1,6],f32>, !torch.vtensor<[2],si32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}
