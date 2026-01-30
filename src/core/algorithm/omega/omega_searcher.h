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

#include <zvec/core/framework/index_framework.h>
#include "../hnsw/hnsw_searcher.h"
#include <omega/omega_api.h>

namespace zvec {
namespace core {

//! OMEGA Index Searcher - extends HNSW with adaptive search
class OmegaSearcher : public HnswSearcher {
 public:
  using ContextPointer = IndexSearcher::Context::Pointer;

 public:
  OmegaSearcher(void);
  ~OmegaSearcher(void);

  OmegaSearcher(const OmegaSearcher &) = delete;
  OmegaSearcher &operator=(const OmegaSearcher &) = delete;

 protected:
  //! Initialize Searcher
  virtual int init(const ailego::Params &params) override;

  //! Cleanup Searcher
  virtual int cleanup(void) override;

  //! Load Index from storage
  virtual int load(IndexStorage::Pointer container,
                   IndexMetric::Pointer metric) override;

  //! Unload index from storage
  virtual int unload(void) override;

  //! KNN Search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          ContextPointer &context) const override {
    return search_impl(query, qmeta, 1, context);
  }

  //! KNN Search with OMEGA adaptive search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          ContextPointer &context) const override;

  // TODO: These methods call protected methods of HnswSearcher and need to be fixed
  /*
  //! Fetch vector by key (delegate to HNSW)
  virtual const void *get_vector(uint64_t key) const override {
    return hnsw_searcher_->get_vector(key);
  }

  //! Create a searcher context (delegate to HNSW)
  virtual ContextPointer create_context() const override {
    return hnsw_searcher_->create_context();
  }

  //! Create a new iterator (delegate to HNSW)
  virtual IndexProvider::Pointer create_provider(void) const override {
    return hnsw_searcher_->create_provider();
  }

  //! Retrieve statistics (delegate to HNSW)
  virtual const Stats &stats(void) const override {
    return hnsw_searcher_->stats();
  }

  //! Retrieve meta of index (delegate to HNSW)
  virtual const IndexMeta &meta(void) const override {
    return hnsw_searcher_->meta();
  }

  //! Retrieve params of index
  virtual const ailego::Params &params(void) const override {
    return params_;
  }

  virtual void print_debug_info() override {
    hnsw_searcher_->print_debug_info();
  }
  */

 private:
  //! Check if OMEGA mode should be used
  bool should_use_omega() const {
    return omega_enabled_ && use_omega_mode_ &&
           omega_model_ != nullptr &&
           omega_model_is_loaded(omega_model_);
  }

  //! Adaptive search with OMEGA predictions
  int adaptive_search(const void *query, const IndexQueryMeta &qmeta,
                      uint32_t count, ContextPointer &context) const;

 private:
  // OMEGA components
  OmegaModelHandle omega_model_;
  bool omega_enabled_;
  bool use_omega_mode_;
  float target_recall_;
  uint32_t min_vector_threshold_;
  size_t current_vector_count_;
  std::string model_dir_;
};

}  // namespace core
}  // namespace zvec
