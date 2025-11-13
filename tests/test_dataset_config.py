"""
Unit tests for dataset configuration validation.

These tests verify that datasets are properly configured without requiring
a live Weaviate instance.
"""

import pytest
import weaviate_datasets as wd
from weaviate.classes.config import Configure, DataType


class TestWineReviews:
    """Test WineReviews dataset configuration."""

    def test_collection_name(self):
        dataset = wd.WineReviews()
        assert dataset.collection_name == "WineReview"

    def test_properties_defined(self):
        dataset = wd.WineReviews()
        assert len(dataset.properties) == 5

        property_names = [prop.name for prop in dataset.properties]
        assert "review_body" in property_names
        assert "title" in property_names
        assert "country" in property_names
        assert "points" in property_names
        assert "price" in property_names

    def test_property_types(self):
        dataset = wd.WineReviews()
        property_types = {prop.name: prop.dataType for prop in dataset.properties}

        assert property_types["review_body"] == DataType.TEXT
        assert property_types["title"] == DataType.TEXT
        assert property_types["country"] == DataType.TEXT
        assert property_types["points"] == DataType.INT
        assert property_types["price"] == DataType.NUMBER

    def test_dataloader_yields_data(self):
        dataset = wd.WineReviews()
        dataloader = dataset._class_dataloader()

        # Get first item
        data_obj, vector = next(dataloader)

        # Check that all expected fields are present
        assert "review_body" in data_obj
        assert "title" in data_obj
        assert "country" in data_obj
        assert "points" in data_obj
        assert "price" in data_obj

        # Vector should be None (not pre-computed)
        assert vector is None

    def test_get_sample(self):
        dataset = wd.WineReviews()
        sample = dataset.get_sample()

        assert isinstance(sample, dict)
        assert "review_body" in sample
        assert "title" in sample


class TestWineReviewsMT:
    """Test WineReviewsMT (multi-tenancy) configuration."""

    def test_collection_name(self):
        dataset = wd.WineReviewsMT()
        assert dataset.collection_name == "WineReviewMT"

    def test_multi_tenancy_enabled(self):
        dataset = wd.WineReviewsMT()
        assert dataset.mt_config is not None

    def test_tenants_defined(self):
        dataset = wd.WineReviewsMT()
        assert len(dataset.tenants) == 2
        tenant_names = [t.name for t in dataset.tenants]
        assert "tenantA" in tenant_names
        assert "tenantB" in tenant_names


class TestWineReviewsNV:
    """Test WineReviewsNV (named vectors) configuration."""

    def test_collection_name(self):
        dataset = wd.WineReviewsNV()
        assert dataset.collection_name == "WineReviewNV"

    def test_named_vectors_configured(self):
        dataset = wd.WineReviewsNV()
        assert isinstance(dataset.vector_config, list)
        assert len(dataset.vector_config) == 3

    def test_vector_names(self):
        dataset = wd.WineReviewsNV()
        vector_names = [vc.name for vc in dataset.vector_config]
        assert "title" in vector_names
        assert "review_body" in vector_names
        assert "title_country" in vector_names


class TestWiki100:
    """Test Wiki100 dataset configuration."""

    def test_collection_name(self):
        dataset = wd.Wiki100()
        assert dataset.collection_name == "WikiChunk"

    def test_custom_collection_name(self):
        dataset = wd.Wiki100(collection_name="CustomWiki")
        assert dataset.collection_name == "CustomWiki"

    def test_properties_defined(self):
        dataset = wd.Wiki100()
        assert len(dataset.properties) == 3

        property_names = [prop.name for prop in dataset.properties]
        assert "title" in property_names
        assert "chunk" in property_names
        assert "chunk_number" in property_names

    def test_default_chunking(self):
        dataset = wd.Wiki100()
        assert dataset.chunking == "wiki_sections"

    def test_set_chunking_methods(self):
        dataset = wd.Wiki100()

        valid_methods = [
            "fixed",
            "wiki_sections",
            "wiki_sections_chunked",
            "wiki_heading_only"
        ]

        for method in valid_methods:
            dataset.set_chunking(method)
            assert dataset.chunking == method

    def test_dataloader_yields_chunks(self):
        dataset = wd.Wiki100()
        dataloader = dataset._class_dataloader()

        # Get first item
        data_obj, vector = next(dataloader)

        assert "title" in data_obj
        assert "chunk" in data_obj
        assert "chunk_number" in data_obj
        assert data_obj["chunk_number"] == 1
        assert vector is None


class TestJeopardyQuestions1k:
    """Test JeopardyQuestions1k dataset configuration."""

    def test_collection_names(self):
        dataset = wd.JeopardyQuestions1k()
        assert dataset._question_collection == "JeopardyQuestion"
        assert dataset._category_collection == "JeopardyCategory"

    def test_uses_existing_vectors_by_default(self):
        dataset = wd.JeopardyQuestions1k()
        assert dataset._use_existing_vecs is True

    def test_custom_vector_config_disables_existing_vecs(self):
        custom_config = Configure.Vectors.text2vec_cohere()
        dataset = wd.JeopardyQuestions1k(vector_config=custom_config)
        assert dataset._use_existing_vecs is False

    def test_dataloader_yields_pairs(self):
        dataset = wd.JeopardyQuestions1k()
        dataloader = dataset._class_pair_dataloader()

        try:
            # Get first pair
            (question_obj, question_vec), (category_obj, category_vec) = next(dataloader)

            # Check question object
            assert "question" in question_obj
            assert "answer" in question_obj
            assert "points" in question_obj
            assert "round" in question_obj
            assert "air_date" in question_obj

            # Check category object
            assert "title" in category_obj

            # Check that vectors exist (pre-computed)
            assert question_vec is not None
            assert category_vec is not None
            assert isinstance(question_vec, list)
            assert isinstance(category_vec, list)
        finally:
            dataloader.close()

    def test_get_sample(self):
        dataset = wd.JeopardyQuestions1k()
        question_sample, category_sample = dataset.get_sample()

        assert isinstance(question_sample, dict)
        assert isinstance(category_sample, dict)
        assert "question" in question_sample
        assert "title" in category_sample


class TestJeopardyQuestions10k:
    """Test JeopardyQuestions10k dataset configuration."""

    def test_inherits_from_1k(self):
        dataset = wd.JeopardyQuestions10k()
        assert dataset._question_collection == "JeopardyQuestion"
        assert dataset._category_collection == "JeopardyCategory"

    def test_uses_different_data_files(self):
        dataset_1k = wd.JeopardyQuestions1k()
        dataset_10k = wd.JeopardyQuestions10k()

        assert dataset_1k._data_fpath != dataset_10k._data_fpath
        assert "10k" in dataset_10k._data_fpath
        assert "1k" in dataset_1k._data_fpath


class TestNewsArticles:
    """Test NewsArticles dataset configuration."""

    def test_collection_names(self):
        dataset = wd.NewsArticles()
        # NewsArticles manages multiple collections
        assert dataset._datadir.endswith("newsarticles")

    def test_embeddings_files_defined(self):
        dataset = wd.NewsArticles()
        assert "articles" in dataset._embeddings_files
        assert "authors" in dataset._embeddings_files
        assert "categories" in dataset._embeddings_files
        assert "publications" in dataset._embeddings_files

    def test_get_sample(self):
        dataset = wd.NewsArticles()
        sample = dataset.get_sample()

        assert isinstance(sample, dict)
        # Should have article properties
        assert "title" in sample or "summary" in sample


class TestDatasetUploadParameters:
    """Test common upload parameters across datasets."""

    def test_wine_reviews_upload_params(self):
        """Test that upload_dataset accepts expected parameters."""
        dataset = wd.WineReviews()

        # These should not raise errors (we're just testing the signature)
        # We won't actually call upload_dataset without a client
        assert hasattr(dataset, "upload_dataset")
        assert callable(dataset.upload_dataset)

    def test_jeopardy_upload_params(self):
        dataset = wd.JeopardyQuestions1k()
        assert hasattr(dataset, "upload_dataset")
        assert callable(dataset.upload_dataset)

    def test_wiki_upload_params(self):
        dataset = wd.Wiki100()
        assert hasattr(dataset, "upload_dataset")
        assert callable(dataset.upload_dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
