// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-backend-to-tosa-linalg-backend-pipeline)' -split-input-file -verify-diagnostics %s | FileCheck %s

//-----

// CHECK-LABEL:   func.func @torch.aten.size.int(
// CHECK-SAME:                                   %[[ARG0:.*]]: tensor<4x2xf32>) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           cf.assert %[[VAL_0]], "dim must be smaller than inputRank"
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[ARG0]], %[[VAL_1]] : tensor<4x2xf32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK:           return %[[VAL_3]] : i64
func.func @torch.aten.size.int(%arg0: !torch.vtensor<[4,2],f32>) -> !torch.int {
    %c2 = torch.constant.int 2
    %0 = torch.aten.size.int %arg0, %c2 : !torch.vtensor<[4,2],f32>, !torch.int -> !torch.int
    return %0 : !torch.int
}

//-----

// CHECK-LABEL:   func.func @tm_scan(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x512xi64>) -> (tensor<1x512xi64>, tensor<1xi64>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 512 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x512xi64>
// CHECK:           %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] : memref<1x512xi64>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1xi64>
// CHECK:           %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<1xi64>
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK:             %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_3]] : index
// CHECK:             scf.if %[[VAL_9]] {
// CHECK:               %[[VAL_10:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_3]], %[[VAL_8]]] : tensor<1x512xi64>
// CHECK:               memref.store %[[VAL_10]], %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_8]]] : memref<1x512xi64>
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : index
// CHECK:               %[[VAL_12:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_11]]] : memref<1x512xi64>
// CHECK:               %[[VAL_13:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_3]], %[[VAL_8]]] : tensor<1x512xi64>
// CHECK:               %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:               memref.store %[[VAL_14]], %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_8]]] : memref<1x512xi64>
// CHECK:               memref.store %[[VAL_14]], %[[VAL_6]]{{\[}}%[[VAL_3]]] : memref<1xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_5]], %[[VAL_7]] : tensor<1x512xi64>, tensor<1xi64>
// CHECK:         }
func.func @tm_scan(%arg0: tensor<1x512xi64>) -> (tensor<1x512xi64>, tensor<1xi64>) {
    %0 = tensor.empty() : tensor<1x512xi64>
    %1 = tensor.empty() : tensor<1xi64>
    %2:2 = tm_tensor.scan dimension(1) inclusive(true) ins(%arg0 : tensor<1x512xi64>) outs(%0, %1 : tensor<1x512xi64>, tensor<1xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %3 = arith.addi %arg1, %arg2 : i64
      tm_tensor.yield %3 : i64
    } -> tensor<1x512xi64>, tensor<1xi64>
    return %2#0, %2#1 : tensor<1x512xi64>, tensor<1xi64>
}