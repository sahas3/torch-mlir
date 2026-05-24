// RUN: not torch-mlir-opt %s -convert-torch-to-tosa --split-input-file 2>&1 | FileCheck %s

// Regression tests for AtenDequantize* lowerings: when the per-tensor
// quantization params (scale / zero_point) are not literal constants but
// rather come from dynamic computations (e.g., aten.item, as emitted by
// the ONNX importer for DynamicQuantizeLinear), the per-channel branch
// of ConvertDequantizeOp used to crash via cast<RankedTensorType>(non-
// tensor type). The fix replaces the crash with a clean
// notifyMatchFailure so the pipeline reports a real diagnostic instead
// of an LLVM assertion.

// CHECK-NOT: PLEASE submit a bug report
// CHECK: failed to legalize operation 'torch.aten.dequantize.self'

func.func @dequantize_self_dynamic_per_tensor_zp(%arg0: !torch.vtensor<[128],ui8>, %arg1: !torch.float, %arg2: !torch.int) -> !torch.vtensor<[128],f32> {
  %0 = torch.aten._make_per_tensor_quantized_tensor %arg0, %arg1, %arg2 : !torch.vtensor<[128],ui8>, !torch.float, !torch.int -> !torch.vtensor<[128],!torch.quint8>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[128],!torch.quint8> -> !torch.vtensor<[128],f32>
  return %1 : !torch.vtensor<[128],f32>
}

// -----

// CHECK-NOT: PLEASE submit a bug report
// CHECK: failed to legalize operation 'torch.aten.dequantize.tensor'

func.func @dequantize_tensor_dynamic_per_tensor_zp(%arg0: !torch.vtensor<[1,128],si8>, %arg1: !torch.float, %arg2: !torch.int) -> !torch.vtensor<[1,128],f32> {
  %0 = torch.aten._make_per_tensor_quantized_tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1,128],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,128],!torch.qint8>
  %1 = torch.aten.dequantize.tensor %0 : !torch.vtensor<[1,128],!torch.qint8> -> !torch.vtensor<[1,128],f32>
  return %1 : !torch.vtensor<[1,128],f32>
}
