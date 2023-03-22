import pytest
from weaviate_datasets.datasets import Dataset
from weaviate_datasets import (
    JeopardyQuestions10k,
    JeopardyQuestions1k,
    WineReviews,
    WikiArticles,
    WikiCities,
)

dataset_classes = [JeopardyQuestions10k, JeopardyQuestions1k, WineReviews, WikiArticles, WikiCities]


def test_instantiation():
    for d in dataset_classes:
        dataset = d()
        assert isinstance(dataset, Dataset)


def test_class_defs_exist():
    for d in dataset_classes:
        dataset = d()
        assert (
            type(dataset.get_class_definitions()) == list
            and len(dataset.get_class_definitions()) > 0
        )


def test_vectorizer_change():
    for d in dataset_classes:
        dataset = d()
        for v in ["text2vec-huggingface", "text2vec-transformer"]:
            new_module_config = {v: "some config"}
            dataset.set_vectorizer(v, new_module_config)
            for c in dataset.get_class_definitions():
                assert c["vectorizer"] == v
                assert c["moduleConfig"] == new_module_config


def test_has_a_dataloader():
    for d in dataset_classes:
        dataset = d()
        assert (
            hasattr(dataset, "_class_dataloader")
            and callable(getattr(dataset, "_class_dataloader"))
        ) or (
            hasattr(dataset, "_class_pair_dataloader")
            and callable(getattr(dataset, "_class_pair_dataloader"))
        )


def test_dataset_size():
    for d in dataset_classes:
        dataset = d()
        assert type(dataset.get_dataset_size()) == int


def test_get_sample():
    for d in dataset_classes:
        dataset = d()
        sample = dataset.get_sample()
        assert type(sample) == dict
        assert len(sample) > 0


# def test_class_dataloader():
#     for d in dataset_classes:
#         hasattr(d, "_class_dataloader") and callable(getattr(d, "_class_dataloader")):

# def test_class_pair_dataloader():
#     for d in dataset_classes:
#         if hasattr(d, "_class_pair_dataloader") and callable(getattr(d, "_class_pair_dataloader")):
