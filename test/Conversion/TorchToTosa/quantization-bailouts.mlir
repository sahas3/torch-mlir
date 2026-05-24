// RUN: not torch-mlir-opt %s -convert-torch-to-tosa --split-input-file 2>&1 | FileCheck %s

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
