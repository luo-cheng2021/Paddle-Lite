// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/x86/clip_compute.h"

REGISTER_LITE_KERNEL(
    clip, kX86, kFloat, kNCHW, paddle::lite::kernels::x86::ClipCompute<float>, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("clip", 1)
    .Finalize();
