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

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <ailego/math/distance.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_builder.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_meta.h"

using namespace std;
using namespace testing;
using namespace zvec::ailego;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

namespace zvec {
namespace core {

constexpr size_t static dim = 16;

class OmegaSearcherTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string OmegaSearcherTest::_dir("OmegaSearcherTest/");
shared_ptr<IndexMeta> OmegaSearcherTest::_index_meta_ptr;

void OmegaSearcherTest::SetUp(void) {
  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void OmegaSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

// Test that OmegaSearcher falls back to HNSW when omega is disabled
TEST_F(OmegaSearcherTest, TestFallbackToHnswWhenDisabled) {
  // Build index using HnswBuilder
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestFallbackToHnswWhenDisabled";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test OmegaSearcher with omega.enabled=false (default)
  IndexSearcher::Pointer omega_searcher =
      IndexFactory::CreateSearcher("OmegaSearcher");
  ASSERT_TRUE(omega_searcher != nullptr);

  // Initialize without enabling omega (should fallback to HNSW)
  ailego::Params params;
  params.insert("omega.enabled", false);  // Explicitly disable omega
  ASSERT_EQ(0, omega_searcher->init(params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, omega_searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = omega_searcher->create_context();
  ASSERT_TRUE(!!ctx);

  // Perform search
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, omega_searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  // Verify results are sorted by distance
  for (size_t k = 1; k < results.size(); ++k) {
    ASSERT_LE(results[k - 1].score(), results[k].score());
  }
}

// Test that OmegaSearcher and HnswSearcher produce identical results when omega is disabled
TEST_F(OmegaSearcherTest, TestIdenticalResultsWithHnsw) {
  // Build index using HnswBuilder
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i + j);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestIdenticalResultsWithHnsw";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Create HnswSearcher
  IndexSearcher::Pointer hnsw_searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(hnsw_searcher != nullptr);
  ASSERT_EQ(0, hnsw_searcher->init(ailego::Params()));

  auto storage1 = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage1->open(path, false));
  ASSERT_EQ(0, hnsw_searcher->load(storage1, IndexMetric::Pointer()));

  // Create OmegaSearcher with omega disabled
  IndexSearcher::Pointer omega_searcher =
      IndexFactory::CreateSearcher("OmegaSearcher");
  ASSERT_TRUE(omega_searcher != nullptr);

  ailego::Params params;
  params.insert("omega.enabled", false);
  ASSERT_EQ(0, omega_searcher->init(params));

  auto storage2 = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage2->open(path, false));
  ASSERT_EQ(0, omega_searcher->load(storage2, IndexMetric::Pointer()));

  // Search with both searchers and compare results
  NumericalVector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = 100.0f + j;
  }

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 20;

  auto hnsw_ctx = hnsw_searcher->create_context();
  hnsw_ctx->set_topk(topk);
  ASSERT_EQ(0, hnsw_searcher->search_impl(query.data(), qmeta, hnsw_ctx));
  auto &hnsw_results = hnsw_ctx->result();

  auto omega_ctx = omega_searcher->create_context();
  omega_ctx->set_topk(topk);
  ASSERT_EQ(0, omega_searcher->search_impl(query.data(), qmeta, omega_ctx));
  auto &omega_results = omega_ctx->result();

  // Results should be identical
  ASSERT_EQ(hnsw_results.size(), omega_results.size());
  for (size_t k = 0; k < hnsw_results.size(); ++k) {
    ASSERT_EQ(hnsw_results[k].key(), omega_results[k].key());
    ASSERT_FLOAT_EQ(hnsw_results[k].score(), omega_results[k].score());
  }
}

// Test OmegaSearcher with RNN search (radius search)
TEST_F(OmegaSearcherTest, TestRnnSearchFallback) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestRnnSearchFallback";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test OmegaSearcher with omega disabled
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("OmegaSearcher");
  ASSERT_TRUE(searcher != nullptr);

  ailego::Params params;
  params.insert("omega.enabled", false);
  ASSERT_EQ(0, searcher->init(params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  // Test with radius threshold
  float radius = results[topk / 2].score();
  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

// Test OmegaSearcher with InnerProduct metric
TEST_F(OmegaSearcherTest, TestInnerProductFallback) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("InnerProduct", 0, ailego::Params());

  ASSERT_EQ(0, builder->init(index_meta, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestInnerProductFallback";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test OmegaSearcher with omega disabled
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("OmegaSearcher");
  ASSERT_TRUE(searcher != nullptr);

  ailego::Params params;
  params.insert("omega.enabled", false);
  ASSERT_EQ(0, searcher->init(params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  // Verify results are sorted correctly for InnerProduct (descending)
  for (size_t k = 1; k < results.size(); ++k) {
    ASSERT_GE(results[k - 1].score(), results[k].score());
  }
}

// Test that omega parameters don't affect HNSW fallback mode
TEST_F(OmegaSearcherTest, TestOmegaParamsIgnoredWhenDisabled) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestOmegaParamsIgnored";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Create two OmegaSearcher instances with different omega params
  // but both with omega disabled
  IndexSearcher::Pointer searcher1 =
      IndexFactory::CreateSearcher("OmegaSearcher");
  ASSERT_TRUE(searcher1 != nullptr);

  ailego::Params params1;
  params1.insert("omega.enabled", false);
  params1.insert("omega.target_recall", 0.95f);
  params1.insert("omega.min_vector_threshold", 10000);
  ASSERT_EQ(0, searcher1->init(params1));

  IndexSearcher::Pointer searcher2 =
      IndexFactory::CreateSearcher("OmegaSearcher");
  ASSERT_TRUE(searcher2 != nullptr);

  ailego::Params params2;
  params2.insert("omega.enabled", false);
  params2.insert("omega.target_recall", 0.85f);
  params2.insert("omega.min_vector_threshold", 5000);
  ASSERT_EQ(0, searcher2->init(params2));

  auto storage1 = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage1->open(path, false));
  ASSERT_EQ(0, searcher1->load(storage1, IndexMetric::Pointer()));

  auto storage2 = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage2->open(path, false));
  ASSERT_EQ(0, searcher2->load(storage2, IndexMetric::Pointer()));

  // Search with both searchers - results should be identical
  // since omega is disabled and both use HNSW
  NumericalVector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = 50.0f;
  }

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 30;

  auto ctx1 = searcher1->create_context();
  ctx1->set_topk(topk);
  ASSERT_EQ(0, searcher1->search_impl(query.data(), qmeta, ctx1));
  auto &results1 = ctx1->result();

  auto ctx2 = searcher2->create_context();
  ctx2->set_topk(topk);
  ASSERT_EQ(0, searcher2->search_impl(query.data(), qmeta, ctx2));
  auto &results2 = ctx2->result();

  // Results should be identical despite different omega params
  ASSERT_EQ(results1.size(), results2.size());
  for (size_t k = 0; k < results1.size(); ++k) {
    ASSERT_EQ(results1[k].key(), results2[k].key());
    ASSERT_FLOAT_EQ(results1[k].score(), results2[k].score());
  }
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
