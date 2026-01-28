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
#pragma once

#include <zvec/core/framework/index_builder.h>
#include "../hnsw/hnsw_builder.h"

namespace zvec {
namespace core {

//! OMEGA Index Builder - wraps HNSW builder
class OmegaBuilder : public IndexBuilder {
 public:
  //! Constructor
  OmegaBuilder();

  //! Initialize the builder
  virtual int init(const IndexMeta &meta,
                   const ailego::Params &params) override;

  //! Cleanup the builder
  virtual int cleanup(void) override;

  //! Train the data (delegate to HNSW)
  virtual int train(IndexThreads::Pointer threads,
                    IndexHolder::Pointer holder) override;

  //! Train the data (delegate to HNSW)
  virtual int train(const IndexTrainer::Pointer &trainer) override;

  //! Build the index (delegate to HNSW)
  virtual int build(IndexThreads::Pointer threads,
                    IndexHolder::Pointer holder) override;

  //! Dump index into storage (delegate to HNSW)
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics (delegate to HNSW)
  virtual const Stats &stats(void) const override {
    return hnsw_builder_->stats();
  }

 private:
  enum BUILD_STATE {
    BUILD_STATE_INIT = 0,
    BUILD_STATE_INITED = 1,
    BUILD_STATE_TRAINED = 2,
    BUILD_STATE_BUILT = 3
  };

  std::shared_ptr<HnswBuilder> hnsw_builder_;
  BUILD_STATE state_{BUILD_STATE_INIT};
};

}  // namespace core
}  // namespace zvec
