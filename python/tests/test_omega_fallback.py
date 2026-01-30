# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    IndexOption,
    MetricType,
    OmegaIndexParam,
    VectorQuery,
    VectorSchema,
)


@pytest.fixture(scope="module", autouse=True)
def init_zvec():
    """Initialize zvec once for all tests in this module."""
    zvec.init()


def test_omega_index_param_creation():
    """Test that OmegaIndexParam can be created with various parameters."""
    # Default parameters
    param1 = OmegaIndexParam()
    assert param1.m == 50
    assert param1.ef_construction == 500
    assert param1.metric_type == MetricType.IP

    # Custom parameters
    param2 = OmegaIndexParam(
        metric_type=MetricType.L2,
        m=32,
        ef_construction=200
    )
    assert param2.m == 32
    assert param2.ef_construction == 200
    assert param2.metric_type == MetricType.L2


def test_omega_collection_creation():
    """Test creating a collection with OMEGA index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_omega_db"

        # Create schema with OMEGA index
        schema = CollectionSchema(
            name="test_omega_collection",
            fields=[
                FieldSchema("id", DataType.INT64, nullable=False),
                FieldSchema("text", DataType.STRING, nullable=False),
            ],
            vectors=[
                VectorSchema(
                    "embedding",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=OmegaIndexParam(
                        metric_type=MetricType.L2,
                        m=16,
                        ef_construction=200
                    ),
                ),
            ],
        )

        # Create collection
        collection = zvec.create_and_open(
            str(db_path),
            schema,
            CollectionOption(read_only=False, enable_mmap=False)
        )

        # Insert some test data
        docs = [
            Doc(
                id=str(i),
                fields={"id": i, "text": f"doc_{i}"},
                vectors={"embedding": np.random.randn(128).astype(np.float32).tolist()}
            )
            for i in range(100)
        ]

        status = collection.insert(docs)
        # insert() returns a list of Status objects for multiple docs
        assert len(status) == len(docs), "Insert returned wrong number of statuses"
        for s in status:
            assert s.ok(), f"Insert failed: {s.message()}"

        # Create index
        collection.create_index(
            field_name="embedding",
            index_param=OmegaIndexParam(metric_type=MetricType.L2, m=16, ef_construction=200),
            option=IndexOption()
        )

        # Query
        query_vector = np.random.randn(128).astype(np.float32).tolist()
        results = collection.query(
            vectors=VectorQuery(field_name="embedding", vector=query_vector),
            topk=10
        )

        assert len(results) > 0, "Query returned no results"
        assert len(results) <= 10, "Query returned more than top_k results"


def test_omega_vs_hnsw_compatibility():
    """Test that OMEGA index produces similar results to HNSW index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path_hnsw = Path(tmpdir) / "test_hnsw_db"
        db_path_omega = Path(tmpdir) / "test_omega_db"

        # Create identical schemas except for index type
        def create_schema(name, index_param):
            return CollectionSchema(
                name=name,
                fields=[
                    FieldSchema("id", DataType.INT64, nullable=False),
                ],
                vectors=[
                    VectorSchema(
                        "embedding",
                        DataType.VECTOR_FP32,
                        dimension=64,
                        index_param=index_param,
                    ),
                ],
            )

        # Create test data
        np.random.seed(42)
        test_docs = [
            Doc(
                id=str(i),
                fields={"id": i},
                vectors={"embedding": np.random.randn(64).astype(np.float32).tolist()}
            )
            for i in range(200)
        ]

        query_vector = np.random.randn(64).astype(np.float32).tolist()

        # Test with HNSW
        schema_hnsw = create_schema(
            "test_hnsw",
            HnswIndexParam(metric_type=MetricType.L2, m=16, ef_construction=200)
        )
        collection_hnsw = zvec.create_and_open(
            str(db_path_hnsw),
            schema_hnsw,
            CollectionOption(read_only=False, enable_mmap=False)
        )
        collection_hnsw.insert(test_docs)
        collection_hnsw.create_index(
            field_name="embedding",
            index_param=HnswIndexParam(metric_type=MetricType.L2, m=16, ef_construction=200),
            option=IndexOption()
        )
        results_hnsw = collection_hnsw.query(
            vectors=VectorQuery(field_name="embedding", vector=query_vector),
            topk=10
        )

        # Test with OMEGA
        schema_omega = create_schema(
            "test_omega",
            OmegaIndexParam(metric_type=MetricType.L2, m=16, ef_construction=200)
        )
        collection_omega = zvec.create_and_open(
            str(db_path_omega),
            schema_omega,
            CollectionOption(read_only=False, enable_mmap=False)
        )
        collection_omega.insert(test_docs)
        collection_omega.create_index(
            field_name="embedding",
            index_param=OmegaIndexParam(metric_type=MetricType.L2, m=16, ef_construction=200),
            option=IndexOption()
        )
        results_omega = collection_omega.query(
            vectors=VectorQuery(field_name="embedding", vector=query_vector),
            topk=10
        )

        # Both should return results
        assert len(results_hnsw) > 0, "HNSW query returned no results"
        assert len(results_omega) > 0, "OMEGA query returned no results"

        # Results should have the same number of documents
        assert len(results_hnsw) == len(results_omega), \
            f"Different number of results: HNSW={len(results_hnsw)}, OMEGA={len(results_omega)}"

        # Verify that OMEGA fallback produces identical results to HNSW
        # Since both use the same index structure (HNSW) with identical parameters,
        # they should return the exact same documents in the same order with the same scores
        for i, (doc_hnsw, doc_omega) in enumerate(zip(results_hnsw, results_omega)):
            assert doc_hnsw.id == doc_omega.id, \
                f"Document ID mismatch at position {i}: HNSW={doc_hnsw.id}, OMEGA={doc_omega.id}"

            # Scores should be identical (or very close due to floating point)
            assert abs(doc_hnsw.score - doc_omega.score) < 1e-5, \
                f"Score mismatch at position {i} for doc {doc_hnsw.id}: " \
                f"HNSW={doc_hnsw.score}, OMEGA={doc_omega.score}, diff={abs(doc_hnsw.score - doc_omega.score)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
