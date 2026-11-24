// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL:   func.func @torchAtenEmbeddingBagPaddingIdx
// CHECK:         %[[VAL_0:.*]]: !torch.vtensor<[1000000,64],f32>
// CHECK:         %[[VAL_1:.*]]: !torch.vtensor<[204790],si64>
// CHECK:         %[[VAL_2:.*]]: !torch.vtensor<[2048],si64>
// CHECK:         %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[2048],si64> -> tensor<2048xi64>
// CHECK:         %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[204790],si64> -> tensor<204790xi64>
// CHECK:         %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1000000,64],f32> -> tensor<1000000x64xf32>
// CHECK-DAG:     %[[VAL_6:.*]] = torch.constant.bool true
// CHECK-DAG:     %[[VAL_7:.*]] = torch.constant.int 0
// CHECK-DAG:     %[[VAL_8:.*]] = torch.constant.bool true
// CHECK:         %[[CHECK_LAST:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : i64
// CHECK:         %[[ZERO_IDX:.*]] = arith.constant 0 : index
// CHECK:         %[[SAFE_IDX:.*]] = arith.select %[[CHECK_LAST]], %[[ZERO_IDX]], {{.*}} : index
// CHECK:         %[[NEXT_OFFSET_RAW:.*]] = tensor.extract %[[VAL_3]][%[[SAFE_IDX]]] : tensor<2048xi64>
// CHECK:         {{.*}} = arith.select %[[CHECK_LAST]], {{.*}}, %[[NEXT_OFFSET_RAW]] : i64
func.func @torchAtenEmbeddingBagPaddingIdx(%weight: !torch.vtensor<[1000000,64],f32>,
                                           %indices: !torch.vtensor<[204790],si64>,
                                           %offsets: !torch.vtensor<[2048],si64>) -> (!torch.vtensor<[2048,64],f32>,
                                                                                     !torch.vtensor<[0],si64>,
                                                                                     !torch.vtensor<[2048],si64>,
                                                                                     !torch.vtensor<[2048],si64>)
 {
    %scale_grad_by_freq = torch.constant.bool true
    %mode = torch.constant.int 0
    %sparse = torch.constant.bool true
    %per_sample_weights = torch.constant.none
    %include_last_offset = torch.constant.bool false
    %padding_idx = torch.constant.none
    %result0, %result1, %result2, %result3 = torch.aten.embedding_bag.padding_idx %weight,
                                                                                  %indices,
                                                                                  %offsets,
                                                                                  %scale_grad_by_freq,
                                                                                  %mode,
                                                                                  %sparse,
                                                                                  %per_sample_weights,
                                                                                  %include_last_offset,
                                                                                  %padding_idx :
                                                                                               !torch.vtensor<[1000000,64],f32>,
                                                                                               !torch.vtensor<[204790],si64>,
                                                                                               !torch.vtensor<[2048],si64>,
                                                                                               !torch.bool,
                                                                                               !torch.int,
                                                                                               !torch.bool,
                                                                                               !torch.none,
                                                                                               !torch.bool,
                                                                                               !torch.none -> !torch.vtensor<[2048,64],f32>,
                                                                                                              !torch.vtensor<[0],si64>,
                                                                                                              !torch.vtensor<[2048],si64>,
                                                                                                              !torch.vtensor<[2048],si64>

    return %result0, %result1, %result2, %result3 : !torch.vtensor<[2048,64],f32>, !torch.vtensor<[0],si64>, !torch.vtensor<[2048],si64>, !torch.vtensor<[2048],si64>
}

// -----

// CHECK: #[[$IDX_MAP:.+]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK: #[[$OFF_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: #[[$OUT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL:   func.func @torchAtenEmbeddingBagPerSampleWeights
// CHECK-DAG:     %[[INDICES:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[7],si64> -> tensor<7xi64>
// CHECK-DAG:     %[[OFFSETS:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[3],si64> -> tensor<3xi64>
// CHECK-DAG:     %[[PSW:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[7],f32> -> tensor<7xf32>
// CHECK:         linalg.generic {indexing_maps = [#[[$IDX_MAP]], #[[$OFF_MAP]], #[[$IDX_MAP]], #[[$OUT_MAP]]]
// CHECK-SAME:      ins(%[[INDICES]], %[[OFFSETS]], %[[PSW]] : tensor<7xi64>, tensor<3xi64>, tensor<7xf32>)
// CHECK:         ^bb0(%{{.*}}: i64, %{{.*}}: i64, %[[SAMPLE_W:.*]]: f32, %[[ACC:.*]]: f32):
// CHECK:           %[[ROW_ELEM:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<10x4xf32>
// CHECK:           %[[SCALED:.*]] = arith.mulf %[[ROW_ELEM]], %[[SAMPLE_W]] : f32
// CHECK:           %[[SUM:.*]] = arith.addf %[[SCALED]], %[[ACC]] : f32
// CHECK:           arith.select %{{.*}}, %[[SUM]], %[[ACC]] : f32
func.func @torchAtenEmbeddingBagPerSampleWeights(%weight: !torch.vtensor<[10,4],f32>,
                                                 %indices: !torch.vtensor<[7],si64>,
                                                 %offsets: !torch.vtensor<[3],si64>,
                                                 %per_sample_weights: !torch.vtensor<[7],f32>) -> !torch.vtensor<[3,4],f32> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %result0, %result1, %result2, %result3 = torch.aten.embedding_bag.padding_idx %weight, %indices, %offsets, %false, %int0, %false, %per_sample_weights, %false, %none : !torch.vtensor<[10,4],f32>, !torch.vtensor<[7],si64>, !torch.vtensor<[3],si64>, !torch.bool, !torch.int, !torch.bool, !torch.vtensor<[7],f32>, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>, !torch.vtensor<[0],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>
  return %result0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torchAtenEmbeddingBagPerSampleWeightsF16
// CHECK:         %[[INIT:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:         %[[BAGS:.*]] = linalg.generic
// CHECK-SAME:      outs(%[[INIT]] : tensor<3x4xf32>)
// CHECK:         ^bb0(%{{.*}}: i64, %{{.*}}: i64, %[[SAMPLE_W:.*]]: f16, %[[ACC:.*]]: f32):
// CHECK:           %[[ROW_ELEM:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<10x4xf16>
// CHECK:           %[[ROW_ELEM_F32:.*]] = arith.extf %[[ROW_ELEM]] : f16 to f32
// CHECK:           %[[SAMPLE_W_F32:.*]] = arith.extf %[[SAMPLE_W]] : f16 to f32
// CHECK:           %[[SCALED:.*]] = arith.mulf %[[ROW_ELEM_F32]], %[[SAMPLE_W_F32]] : f32
// CHECK:           %[[SUM:.*]] = arith.addf %[[SCALED]], %[[ACC]] : f32
// CHECK:         } -> tensor<3x4xf32>
// CHECK:         linalg.generic {{.*}} ins(%[[BAGS]] : tensor<3x4xf32>) outs(%{{.*}} : tensor<3x4xf16>)
// CHECK:           arith.truncf %{{.*}} : f32 to f16
func.func @torchAtenEmbeddingBagPerSampleWeightsF16(%weight: !torch.vtensor<[10,4],f16>,
                                                    %indices: !torch.vtensor<[7],si64>,
                                                    %offsets: !torch.vtensor<[3],si64>,
                                                    %per_sample_weights: !torch.vtensor<[7],f16>) -> !torch.vtensor<[3,4],f16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %result0, %result1, %result2, %result3 = torch.aten.embedding_bag.padding_idx %weight, %indices, %offsets, %false, %int0, %false, %per_sample_weights, %false, %none : !torch.vtensor<[10,4],f16>, !torch.vtensor<[7],si64>, !torch.vtensor<[3],si64>, !torch.bool, !torch.int, !torch.bool, !torch.vtensor<[7],f16>, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f16>, !torch.vtensor<[0],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>
  return %result0 : !torch.vtensor<[3,4],f16>
}
