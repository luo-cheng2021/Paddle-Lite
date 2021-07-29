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

#include "lite/kernels/x86/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <limits>
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/gen_code/gen_code.h"
#include "lite/model_parser/model_parser.h"
#include "lite/model_parser/pb/program_desc.h"
#include <frontend_manager/frontend_manager.hpp>

namespace paddle {
namespace lite {
namespace kernels {
namespace ov {

void AddFeed(cpp::ProgramDesc& desc, int block_idx, Scope& scope, const std::string& in_name, int col = 0)
{
  auto* block = desc.GetBlock<cpp::BlockDesc>(block_idx);
  auto* op = block->AddOp<cpp::OpDesc>();

  op->SetType("feed");
  op->SetInput("X", {"feed"});
  op->SetOutput("Out", {in_name});
  op->SetAttr("col", col);

  auto y = block->AddVar<cpp::VarDesc>();
  y->SetName("feed");
  y->SetPersistable(true);
  y->SetDataType(VarDataType::BOOL);
  y->SetType(VarDataType::FEED_MINIBATCH);
}

void AddFetch(cpp::ProgramDesc& desc, int block_idx, Scope& scope, const std::string& in_name, int col = 0)
{
  auto* block = desc.GetBlock<cpp::BlockDesc>(block_idx);
  auto* op = block->AddOp<cpp::OpDesc>();

  op->SetType("fetch");
  op->SetInput("X", {in_name});
  op->SetOutput("Out", {"fetch"});
  op->SetAttr("col", col);

  auto y = block->AddVar<cpp::VarDesc>();
  y->SetName("fetch");
  y->SetPersistable(true);
  y->SetDataType(VarDataType::BOOL);
  y->SetType(VarDataType::FETCH_LIST);
}

#define MODEL_TMP_DIR "_ov_tmp"
#define MODEL_TMP_PATH MODEL_TMP_DIR "/model.pdmodel"

using namespace InferenceEngine;

void SubgraphEngine::Serialize() 
{
    std::shared_ptr<cpp::ProgramDesc> new_program = std::make_shared<cpp::ProgramDesc>();
    new_program->CopyFrom(*program_desc_);
    origin_program_->SaveRuntimProgramIntoProgramDesc(new_program);
    
    for (size_t i = 0; i < input_names_.size(); i++)
    {
      AddFeed(*new_program, block_idx_, *exec_scope_, input_names_[i], (int)i);
    }
    for (size_t i = 0; i < output_names_.size(); i++)
    {
      AddFetch(*new_program, block_idx_, *exec_scope_, output_names_[i], (int)i);
    }
    SaveModelPb(MODEL_TMP_DIR, *exec_scope_, *new_program, true);

    // framework::proto::ProgramDesc pb_proto_desc;
    // lite::pb::ProgramDesc pb_desc(&pb_proto_desc);
    // lite::TransformProgramDescCppToAny(*new_program, &pb_desc);

    // std::ofstream file("xxx.pdmodel");
    // file << pb_proto_desc.SerializeAsString();
    // file.close();
}

void SubgraphEngine::ConvertAndCreateIe() 
{
  ngraph::frontend::FrontEndManager manager;
  auto fe = manager.load_by_framework("pdpd");
  auto inputModel = fe->load_from_file(MODEL_TMP_PATH);
  auto ngFunc = fe->convert(inputModel);
  CNNNetwork network(ngFunc);
  executable_network_ = core_.LoadNetwork(network, "CPU");
  infer_request_ = executable_network_.CreateInferRequest();
}

bool SubgraphEngine::BuildDeviceProgram() 
{
  if (!origin_program_) 
  {
    BuildOriginProgram();
  }

  const auto& insts = origin_program_->instructions(kRootBlockIdx);
  for (auto& inst : insts) 
  {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();

    //if (subgraph::CHECK_FAILED(status)) 
    //{
    //  return false;
    //}
  }

  // TODO: pass by object luocheng
  // serialize model
  Serialize();
  
  // convert model and create inference engine
  ConvertAndCreateIe();

  // alloc output memory
  for (auto i = 0; i < origin_otensors_.size(); i++) 
  {
    auto type = origin_otensors_[i]->precision();
    switch (type) 
    {
      case PrecisionType::kFloat:
        origin_otensors_[i]->mutable_data<float>();
        break;
      case PrecisionType::kInt8:
      case PrecisionType::kUInt8:
        origin_otensors_[i]->mutable_data<int8_t>();
        break;
      case PrecisionType::kInt16:
        origin_otensors_[i]->mutable_data<int16_t>();
        break;
      case PrecisionType::kInt32:
        origin_otensors_[i]->mutable_data<int32_t>();
        break;
      case PrecisionType::kInt64:
        origin_otensors_[i]->mutable_data<int64_t>();
        break;
      default:
        LOG(FATAL) << "[OpenVINO] can't mutable data with precision type "
                   << static_cast<int>(type);
        break;
    }
  }

  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() 
{
  // Set input buffer
  for (size_t i = 0; i < origin_itensors_.size(); i++)
  {
    Blob::Ptr output = infer_request_.GetBlob(input_names_[i]);
    auto mem_output = as<MemoryBlob>(output);
    memcpy(mem_output->wmap().as<void *>(), origin_itensors_[i]->raw_data(), origin_itensors_[i]->memory_size());
  }
  infer_request_.Infer();
  // Set output buffer
  for (size_t i = 0; i < origin_otensors_.size(); i++)
  {
    Blob::Ptr output = infer_request_.GetBlob(output_names_[i]);
    auto mem_output = as<MemoryBlob>(output);
    memcpy(origin_otensors_[i]->raw_data(), mem_output->rmap().as<void *>(), origin_otensors_[i]->memory_size());
  }

  return true;
}

void SubgraphCompute::PrepareForRun() 
{
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.block_idx,
                                   param.program_desc,
                                   param.exec_scope,
                                   param.input_data_names,
                                   param.output_data_names));
  CHECK(engine_);
}

void SubgraphCompute::Run() 
{
  CHECK(engine_);
  engine_->Run();
}

}  // namespace imagination_nna
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::ov::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();
