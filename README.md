## Weaviate demo data uploader

This aims to make it easy to upload demo data to your instance of Weaviate for learning how to use it. 

## Usage

All datasets are based on `Dataset` superclass, and includes a number of built-in methods to make it easier to work with it. 

Once you instantiate a dataset, to upload it to Weaviate the syntax is as follows:

```python
import wv_datasets
dataset = wv_datasets.JeopardyQuestions()
dataset.add_to_schema(client)
dataset.upload_objects(client, batch_size=100)
```

Where `client` is the instantiated `weaviate.Client` object.

### Built-in methods

- `.see_class_definitions()`: See the schema definition to be added
- `.classes_in_schema()`: Check whether each class is already in the Weaviate schema
- `.get_class_names()`: Get class names in the dataset

## Available classes

- WikiArticles 
- WineReviews
- JeopardyQuestions
