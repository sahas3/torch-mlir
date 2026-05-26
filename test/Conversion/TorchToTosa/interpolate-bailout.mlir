// RUN: not torch-mlir-opt %s -convert-torch-to-tosa --split-input-file 2>&1 | FileCheck %s

// CHECK-NOT: PLEASE submit a bug report
// CHECK: failed to legalize operation 'torch.aten.__interpolate.size_list_scale_list'

func.func @interpolate_dynamic_spatial_dims(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %none = torch.constant.none
  %str = torch.constant.str "bilinear"
  %false = torch.constant.bool false
  %fp = torch.constant.float 2.0
  %0 = torch.prim.ListConstruct %fp, %fp : (!torch.float, !torch.float) -> !torch.list<float>
  %1 = torch.aten.__interpolate.size_list_scale_list %arg0, %none, %0, %str, %false, %none, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.none, !torch.list<float>, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %1 : !torch.vtensor<[?,?,?,?],f32>
}
