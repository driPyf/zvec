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
#include "../hnsw/hnsw_streamer.h"

namespace zvec {
namespace core {

//! OMEGA Index Streamer - wraps HNSW streamer
class OmegaStreamer : public IndexStreamer {
 public:
  using ContextPointer = IndexStreamer::Context::Pointer;

  OmegaStreamer(void);
  virtual ~OmegaStreamer(void);

  OmegaStreamer(const OmegaStreamer &streamer) = delete;
  OmegaStreamer &operator=(const OmegaStreamer &streamer) = delete;

 protected:
  //! Initialize Streamer
  virtual int init(const IndexMeta &imeta,
                   const ailego::Params &params) override;

  //! Cleanup Streamer
  virtual int cleanup(void) override;

  //! Create a context (delegate to HNSW)
  virtual Context::Pointer create_context(void) const override {
    return hnsw_streamer_->create_context();
  }

  //! Create a new iterator (delegate to HNSW)
  virtual IndexProvider::Pointer create_provider(void) const override {
    return hnsw_streamer_->create_provider();
  }

  //! Add a vector into index (delegate to HNSW)
  virtual int add_impl(uint64_t pkey, const void *query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &context) override {
    return hnsw_streamer_->add_impl(pkey, query, qmeta, context);
  }

  //! Add a vector with id into index (delegate to HNSW)
  virtual int add_with_id_impl(uint32_t id, const void *query,
                               const IndexQueryMeta &qmeta,
                               Context::Pointer &context) override {
    return hnsw_streamer_->add_with_id_impl(id, query, qmeta, context);
  }

  //! Similarity search (delegate to HNSW)
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          Context::Pointer &context) const override {
    return hnsw_streamer_->search_impl(query, qmeta, context);
  }

  //! Similarity search (delegate to HNSW)
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          Context::Pointer &context) const override {
    return hnsw_streamer_->search_impl(query, qmeta, count, context);
  }

  //! Similarity brute force search (delegate to HNSW)
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             Context::Pointer &context) const override {
    return hnsw_streamer_->search_bf_impl(query, qmeta, context);
  }

  //! Similarity brute force search (delegate to HNSW)
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count,
                             Context::Pointer &context) const override {
    return hnsw_streamer_->search_bf_impl(query, qmeta, count, context);
  }

  //! Linear search by primary keys (delegate to HNSW)
  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, ContextPointer &context) const override {
    return hnsw_streamer_->search_bf_by_p_keys_impl(query, p_keys, qmeta,
                                                     context);
  }

  //! Linear search by primary keys (delegate to HNSW)
  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, uint32_t count,
      ContextPointer &context) const override {
    return hnsw_streamer_->search_bf_by_p_keys_impl(query, p_keys, qmeta,
                                                     count, context);
  }

  //! Remove a vector from index (delegate to HNSW)
  virtual int remove_impl(uint64_t pkey, Context::Pointer &context) override {
    return hnsw_streamer_->remove_impl(pkey, context);
  }

  //! Fetch vector by key (delegate to HNSW)
  virtual const void *get_vector(uint64_t key) const override {
    return hnsw_streamer_->get_vector(key);
  }

  //! Retrieve statistics (delegate to HNSW)
  virtual const Stats &stats(void) const override {
    return hnsw_streamer_->stats();
  }

  //! Retrieve meta of index (delegate to HNSW)
  virtual const IndexMeta &meta(void) const override {
    return hnsw_streamer_->meta();
  }

  //! Retrieve params of index
  virtual const ailego::Params &params(void) const override {
    return params_;
  }

  virtual void print_debug_info() override {
    hnsw_streamer_->print_debug_info();
  }

 private:
  std::shared_ptr<HnswStreamer> hnsw_streamer_;
  ailego::Params params_{};
};

}  // namespace core
}  // namespace zvec
