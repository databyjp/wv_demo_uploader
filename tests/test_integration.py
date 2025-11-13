"""
Integration tests for dataset upload and retrieval.

These tests require a live Weaviate instance and will actually upload datasets
and retrieve objects to verify they have the correct shape.

Prerequisites:
- Local Weaviate instance running (default: localhost:8080)
- OPENAI_APIKEY environment variable set (for vectorization)

Run with: pytest tests/test_integration.py -v
"""

import pytest
import weaviate
import weaviate_datasets as wd
import os
from weaviate import WeaviateClient
from typing import Dict, Any
from weaviate_datasets.datasets import SimpleDataset
from weaviate.classes.query import QueryReference


@pytest.fixture(scope="module")
def weaviate_client():
    """
    Create a Weaviate client for testing.
    Skip all tests if Weaviate is not available.
    """
    try:
        client = weaviate.connect_to_local(
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY", "dummy-key")}
        )
        # Test connection
        if not client.is_ready():
            pytest.skip("Weaviate instance is not ready")
        yield client
        client.close()
    except Exception as e:
        pytest.skip(f"Could not connect to Weaviate: {e}")


class TestDatasetIntegration:
    """Integration tests that upload and retrieve actual data from Weaviate."""

    @pytest.mark.parametrize(
        "dataset_class,collection_name,expected_fields,sample_count",
        [
            (
                wd.WineReviews,
                "WineReview",
                ["review_body", "title", "country", "points", "price"],
                3,
            ),
            (
                wd.WineReviewsMT,
                "WineReviewMT",
                ["review_body", "title", "country", "points", "price"],
                3,
            ),
            (
                wd.WineReviewsNV,
                "WineReviewNV",
                ["review_body", "title", "country", "points", "price"],
                3,
            ),
            # # WikiChunk collection not tested as unused
            # (
            #     wd.Wiki100,
            #     "WikiChunk",
            #     ["title", "chunk", "chunk_number"],
            #     5,
            # ),
        ],
    )
    def test_single_collection_upload_and_retrieve(
        self,
        weaviate_client: WeaviateClient,
        dataset_class,
        collection_name: str,
        expected_fields: list,
        sample_count: int,
    ):
        """
        Test uploading a single-collection dataset and retrieving objects.
        """
        # Instantiate and upload dataset
        dataset: SimpleDataset = dataset_class()
        weaviate_client.collections.delete(collection_name)  # Clean up before test
        dataset.upload_dataset(client=weaviate_client, batch_size=50, overwrite=True)

        # Get the collection
        collection = weaviate_client.collections.get(collection_name)

        # Verify collection exists and has data
        assert collection is not None

        # For multi-tenancy datasets, we need to query with tenant
        if hasattr(dataset, "mt_config") and dataset.mt_config is not None:
            # Test with first tenant
            tenant_name = dataset.tenants[0].name
            tenant_collection = collection.with_tenant(tenant_name)
            response = tenant_collection.query.fetch_objects(limit=sample_count, include_vector=True)
        else:
            # Regular collection
            response = collection.query.fetch_objects(limit=sample_count, include_vector=True)

        # Verify we got objects back
        assert len(response.objects) > 0
        assert len(response.objects) <= sample_count

        # Verify each object has the expected fields
        for obj in response.objects:
            properties = obj.properties
            assert isinstance(properties, dict)

            # Check all expected fields are present
            for field in expected_fields:
                assert field in properties, f"Field '{field}' missing from object"

            # For named vectors, verify multiple vectors exist
            if isinstance(dataset, wd.WineReviewsNV):
                assert obj.vector is not None
                assert isinstance(obj.vector, dict)
                # Should have 3 named vectors
                assert len(obj.vector) == 3
                assert "title" in obj.vector
                assert "review_body" in obj.vector
                assert "title_country" in obj.vector

    @pytest.mark.parametrize(
        "dataset_class,question_collection,category_collection,question_fields,category_fields,sample_count",
        [
            (
                wd.JeopardyQuestions1k,
                "JeopardyQuestion",
                "JeopardyCategory",
                ["question", "answer", "points", "round", "air_date"],
                ["title"],
                5,
            ),
        ],
    )
    def test_multi_collection_upload_and_retrieve(
        self,
        weaviate_client: WeaviateClient,
        dataset_class,
        question_collection: str,
        category_collection: str,
        question_fields: list,
        category_fields: list,
        sample_count: int,
    ):
        """
        Test uploading multi-collection datasets and retrieving objects with references.
        """
        # Instantiate and upload dataset
        dataset = dataset_class()
        dataset.upload_dataset(weaviate_client, batch_size=50, overwrite=True)

        # Get both collections
        questions = weaviate_client.collections.get(question_collection)
        categories = weaviate_client.collections.get(category_collection)

        assert questions is not None
        assert categories is not None

        # Query question objects
        question_response = questions.query.fetch_objects(
            limit=sample_count, include_vector=True
        )

        assert len(question_response.objects) > 0
        assert len(question_response.objects) <= sample_count

        # Verify question objects have correct fields
        for obj in question_response.objects:
            properties = obj.properties
            assert isinstance(properties, dict)

            # Check all expected fields
            for field in question_fields:
                assert field in properties, f"Field '{field}' missing from question"

            # Verify vector exists (pre-computed embeddings)
            assert obj.vector is not None
            # OpenAI ada-002 embeddings are 1536 dimensions
            assert len(obj.vector["default"]) == 1536

        # Query category objects
        category_response = categories.query.fetch_objects(
            limit=sample_count, include_vector=True
        )

        assert len(category_response.objects) > 0

        # Verify category objects have correct fields
        for obj in category_response.objects:
            properties = obj.properties
            assert isinstance(properties, dict)

            for field in category_fields:
                assert field in properties, f"Field '{field}' missing from category"

            # Verify vector exists
            assert obj.vector is not None
            assert len(obj.vector["default"]) == 1536

        # Test cross-reference: fetch a question with its category reference
        question_with_ref = questions.query.fetch_objects(
            limit=1, return_references=[QueryReference(link_on="hasCategory", return_properties=["title"])]
        )

        assert len(question_with_ref.objects) > 0
        first_question = question_with_ref.objects[0]

        # Verify the reference exists
        if hasattr(first_question, "references") and first_question.references:
            has_category = first_question.references.get("hasCategory")
            if has_category and len(has_category.objects) > 0:
                category_ref = has_category.objects[0]
                assert "title" in category_ref.properties


    @pytest.mark.parametrize(
        "dataset_class,collection_name,compress",
        [
            (wd.WineReviews, "WineReview_compressed", True),
            (wd.WineReviews, "WineReview_uncompressed", False),
        ],
    )
    def test_compression_options(
        self,
        weaviate_client: WeaviateClient,
        dataset_class,
        collection_name: str,
        compress: bool,
    ):
        """
        Test that datasets can be uploaded with and without compression.
        """
        dataset = dataset_class(collection_name=collection_name)
        dataset.upload_dataset(
            weaviate_client, batch_size=50, overwrite=True, compress=compress
        )

        collection = weaviate_client.collections.get(collection_name)
        response = collection.query.fetch_objects(limit=3)

        assert len(response.objects) > 0

        for obj in response.objects:
            # Verify data integrity regardless of compression
            assert "title" in obj.properties
            assert "review_body" in obj.properties

    def test_get_sample_matches_uploaded_data(self, weaviate_client: WeaviateClient):
        """
        Test that get_sample() returns data matching what's uploaded.
        """
        dataset = wd.WineReviews(collection_name="WineReview_sample_test")
        sample = dataset.get_sample()

        # Upload dataset
        dataset.upload_dataset(weaviate_client, batch_size=50, overwrite=True)

        # Retrieve objects
        collection = weaviate_client.collections.get("WineReview_sample_test")
        response = collection.query.fetch_objects(limit=50)

        # Verify sample has same structure as uploaded objects
        assert len(response.objects) > 0
        first_obj = response.objects[0].properties

        # Same fields should exist
        for field in sample.keys():
            assert field in first_obj, f"Sample field '{field}' not in uploaded data"

        for field in first_obj.keys():
            assert field in sample, f"Uploaded field '{field}' not in sample"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
