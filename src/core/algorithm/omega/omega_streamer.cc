// Copyright 2025-present the zvec project
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

#include "omega_streamer.h"
#include <aitheta2/index_framework.h>

namespace zvec {
namespace core {

OmegaStreamer::OmegaStreamer(void) : hnsw_streamer_(nullptr) {}

OmegaStreamer::~OmegaStreamer(void) {
  this->cleanup();
}

int OmegaStreamer::init(const IndexMeta &imeta, const ailego::Params &params) {
  params_ = params;

  // Create underlying HNSW streamer
  hnsw_streamer_ = std::make_shared<HnswStreamer>();
  int ret = hnsw_streamer_->init(imeta, params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize HNSW streamer");
    return ret;
  }

  LOG_INFO("OmegaStreamer initialized");
  return 0;
}

int OmegaStreamer::cleanup(void) {
  if (hnsw_streamer_ != nullptr) {
    hnsw_streamer_->cleanup();
    hnsw_streamer_.reset();
  }
  return 0;
}

}  // namespace core
}  // namespace zvec

INDEX_FACTORY_REGISTER_STREAMER(zvec::core::OmegaStreamer);
