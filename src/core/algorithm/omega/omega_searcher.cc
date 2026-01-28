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

#include "omega_searcher.h"
#include <aitheta2/index_framework.h>

namespace zvec {
namespace core {

OmegaSearcher::OmegaSearcher(void)
    : hnsw_searcher_(nullptr),
      omega_model_(nullptr),
      omega_enabled_(false),
      use_omega_mode_(false),
      target_recall_(0.95f),
      min_vector_threshold_(10000),
      current_vector_count_(0) {}

OmegaSearcher::~OmegaSearcher(void) {
  this->cleanup();
}

int OmegaSearcher::init(const ailego::Params &params) {
  if (state_ != STATE_INIT) {
    LOG_ERROR("OmegaSearcher already initialized");
    return PROXIMA_BE_ERROR_CODE(DuplicateInit);
  }

  params_ = params;

  // Get OMEGA-specific parameters
  omega_enabled_ = params.get_as_bool("omega.enabled", false);
  target_recall_ = params.get_as_float("omega.target_recall", 0.95f);
  min_vector_threshold_ = params.get_as_uint32("omega.min_vector_threshold", 10000);
  model_dir_ = params.get_as_string("omega.model_dir", "");

  // Create underlying HNSW searcher
  hnsw_searcher_ = std::make_shared<HnswSearcher>();
  int ret = hnsw_searcher_->init(params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize HNSW searcher");
    return ret;
  }

  state_ = STATE_INITED;
  LOG_INFO("OmegaSearcher initialized (omega_enabled=%d, target_recall=%.2f, "
           "min_threshold=%u)",
           omega_enabled_, target_recall_, min_vector_threshold_);
  return 0;
}

int OmegaSearcher::cleanup(void) {
  if (state_ == STATE_INIT) {
    return 0;
  }

  // Cleanup OMEGA model
  if (omega_model_ != nullptr) {
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
  }

  // Cleanup HNSW searcher
  if (hnsw_searcher_ != nullptr) {
    hnsw_searcher_->cleanup();
    hnsw_searcher_.reset();
  }

  state_ = STATE_INIT;
  return 0;
}

int OmegaSearcher::load(IndexStorage::Pointer container,
                        IndexMetric::Pointer metric) {
  if (state_ != STATE_INITED) {
    LOG_ERROR("OmegaSearcher not initialized");
    return PROXIMA_BE_ERROR_CODE(InvalidState);
  }

  // Load HNSW index
  int ret = hnsw_searcher_->load(container, metric);
  if (ret != 0) {
    LOG_ERROR("Failed to load HNSW index");
    return ret;
  }

  // Get vector count from HNSW stats
  current_vector_count_ = hnsw_searcher_->stats().total_doc_count;

  // Try to load OMEGA model if enabled and threshold met
  use_omega_mode_ = false;
  if (omega_enabled_ && current_vector_count_ >= min_vector_threshold_) {
    if (!model_dir_.empty()) {
      omega_model_ = omega_model_create();
      if (omega_model_ != nullptr) {
        ret = omega_model_load(omega_model_, model_dir_.c_str());
        if (ret == 0 && omega_model_is_loaded(omega_model_)) {
          use_omega_mode_ = true;
          LOG_INFO("OMEGA model loaded successfully from %s", model_dir_.c_str());
        } else {
          LOG_WARN("Failed to load OMEGA model from %s, falling back to HNSW",
                   model_dir_.c_str());
          omega_model_destroy(omega_model_);
          omega_model_ = nullptr;
        }
      }
    } else {
      LOG_WARN("OMEGA enabled but model_dir not specified, falling back to HNSW");
    }
  } else {
    if (omega_enabled_) {
      LOG_INFO("Vector count (%zu) below threshold (%u), using standard HNSW",
               current_vector_count_, min_vector_threshold_);
    }
  }

  state_ = STATE_LOADED;
  return 0;
}

int OmegaSearcher::unload(void) {
  if (state_ != STATE_LOADED) {
    return 0;
  }

  // Unload OMEGA model
  if (omega_model_ != nullptr) {
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
  }
  use_omega_mode_ = false;

  // Unload HNSW index
  if (hnsw_searcher_ != nullptr) {
    hnsw_searcher_->unload();
  }

  state_ = STATE_INITED;
  return 0;
}

int OmegaSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               uint32_t count,
                               ContextPointer &context) const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("OmegaSearcher not loaded");
    return PROXIMA_BE_ERROR_CODE(InvalidState);
  }

  // If OMEGA mode is not active, delegate to HNSW
  if (!should_use_omega()) {
    return hnsw_searcher_->search_impl(query, qmeta, count, context);
  }

  // TODO: Implement adaptive search with OMEGA
  // For now, just delegate to HNSW
  // In the future, this will:
  // 1. Create OmegaSearchHandle
  // 2. Perform search with dynamic EF adjustment
  // 3. Use early stopping based on model predictions
  LOG_DEBUG("OMEGA adaptive search not yet implemented, using HNSW");
  return hnsw_searcher_->search_impl(query, qmeta, count, context);
}

}  // namespace core
}  // namespace zvec

INDEX_FACTORY_REGISTER_SEARCHER(zvec::core::OmegaSearcher);
