# Train a Sentence Transformer model for news in spanish

## Fine tune and evaluate a sentence transformer model in Spanish news and upload it to Huggingface Hub

This project is a tutorial about how to train or finetune a sentence transformer model for spanish language. For this exercise, we will use a very helpful library, SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in the paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).

This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images. The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and achieve state-of-the-art performance in various task. Text is embedding in vector space such that similar text is close and can efficiently be found using cosine similarity. The framework provides a large list of Pretrained Models for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases. Pre-trained models can be loaded by just passing the model name: `SentenceTransformer('model_name')`. Then it allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

You can use Sentence Transformer for:

- Computing Sentence Embeddings
- Semantic Textual Similarity
- Clustering
- Paraphrase Mining
- Translated Sentence Mining
- Semantic Search
- Retrieve & Re-Rank
- Text Summarization
- Multilingual Image Search, Clustering & Duplicate Detection

## Model Description

Our pretrained model is [Bertin](https://huggingface.co/bertin-project/bertin-roberta-base-spanish), a BERT-based model in spanish. For this exercise, we went throught trail and error and tested other models like RuPERTa or BETO.

## Problem description
We will train, fine-tune, a BERT based model called BERTIN to generate text embeddings. Using an especialized dataset on news.

Following steps will be developed: 
 
1. Create an Experiment and Trial to keep track of our experiments

2. Load the training data to our training instance

3. Create the scripts to train our custom model, a Transformer.

4. Create an Estimator to train our model in a Tensorflow 2.1 container in script mode

5. Create metric definitions to keep track of them in SageMaker

4. Download the trained model to make predictions

5. Resume training using the latest checkpoint from a previous training 

## The data set

We are using a dataset of IMDB reviews in spanish. After preprocessing, the dataset will be uploaded to our `sagemaker_session_bucket` to be used within our training job. 

## Content

There several notebooks:
- Sentence_transformer_train_sm: Code to train the model in Amazon Sagemaker
- Train and evaluate Sentence transformer spanish: Colab notebook to train in Colab.
- sentence-transformers-data-proc-sm: Data processing
- Sentence_transformer_fromS3_to_hub: Code to upload the model to Hugging Face hub from S3.

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License.