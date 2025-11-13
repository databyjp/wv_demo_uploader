# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational Python package (`weaviate-demo-datasets`) that provides pre-configured demo datasets for uploading to Weaviate vector database instances. The target audience is developers learning Weaviate.

## Development Commands

### Building and Publishing
```bash
# Build the package
python -m build

# Publish to PyPI (requires twine)
twine upload dist/*
```

### Testing
```bash
# Run tests
pytest

# Test imports manually
python tests/test_imports.py  # Requires local Weaviate instance and OPENAI_APIKEY env var
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Core Design Pattern

The codebase uses a class-based hierarchy for dataset definitions:

1. **`SimpleDataset`** (base class in `weaviate_datasets/datasets.py:106-218`):
   - Provides core functionality for single-collection datasets
   - Handles collection creation, batch uploading, and vector index configuration
   - Key methods: `add_collection()`, `upload_objects()`, `upload_dataset()`, `get_sample()`
   - Uses generator pattern via `_class_dataloader()` for memory-efficient data loading

2. **Dataset-specific classes** inherit from `SimpleDataset`:
   - **`WineReviews`**: Simple wine review dataset (50 items)
   - **`WineReviewsMT`**: Multi-tenancy variant with tenants A and B
   - **`WineReviewsNV`**: Named vectors variant with three separate vector configurations
   - **`Wiki100`**: Wikipedia articles with configurable chunking strategies

3. **Complex datasets** (`JeopardyQuestions1k/10k`, `NewsArticles`) implement their own patterns:
   - Handle multiple collections with cross-references
   - Use pre-computed embeddings (stored as .npy and .json files)
   - Implement custom batch loading logic with reference handling

### Data Loading Pattern

All datasets follow a common loading pattern:
1. Override `_class_dataloader()` generator to yield `(data_obj, vector)` tuples
2. `upload_objects()` uses batch uploading with configurable batch size (default: 200)
3. `upload_dataset()` orchestrates: delete (if overwrite), create collection, upload objects
4. Vector compression can be enabled with `compress=True` parameter (uses Binary Quantization)

### Cross-Reference Handling

For datasets with multiple collections (Jeopardy, NewsArticles):
- Collections must be created in dependency order
- References added after all objects exist
- Use `generate_uuid5()` for deterministic UUIDs based on object properties
- `JeopardyQuestions` uses `_class_pair_dataloader()` to yield both question and category objects

### Chunking Strategies (Wiki100)

Four chunking methods via `set_chunking()`:
- `wiki_sections`: Parse by Wikipedia section headings (default)
- `wiki_sections_chunked`: Parse sections then chunk to 200 chars
- `wiki_heading_only`: Extract only section headings
- `fixed`: Fixed 200-char chunks with 20-char overlap

## Key Implementation Details

### Vector Configuration
- Default vectorizer: `text2vec-openai`
- Supports custom vector configs via constructor parameters
- Named vectors supported (see `WineReviewsNV`)
- Compression via Binary Quantization available with `compress=True`

### Multi-Tenancy
- Enable via `Configure.multi_tenancy(enabled=True)`
- Requires tenant list: `[Tenant(name="tenantA"), Tenant(name="tenantB")]`
- Batch operations iterate over tenants using `collection.with_tenant()`

### NewsArticles Dataset
- Downloads 60MB zip file on first use from GitHub
- Four collections: Article, Author, Publication, Category
- Complex bidirectional references between all collections
- References must be added after collection creation using `config.add_reference()`

### Data Files Location
All data files in `weaviate_datasets/data/`:
- Jeopardy: JSON + NPY (embeddings) + CSV (category embeddings)
- Wine: CSV file (`winemag_tiny.csv`)
- Wiki: 100 individual .txt files in `wiki100/` subdirectory
- NewsArticles: Lazy-loaded from zip file, extracted on demand

## Package Structure

```
weaviate_datasets/
├── __init__.py          # Exports public dataset classes
├── datasets.py          # All dataset implementations
└── data/                # Embedded data files (excluded from git via patterns)
```

## Important Notes

- Package includes pre-computed OpenAI embeddings to avoid re-embedding costs
- `MANIFEST.in` controls which data files are included in distribution
- NewsArticles data files are downloaded at runtime (too large for package)
- All datasets use `generate_uuid5()` for deterministic object IDs
- Batch operations use `tqdm` for progress tracking
- When using existing vectors, set `_use_existing_vecs = True` (default for default configs)
