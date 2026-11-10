# Deep Learning with TensorFlow

Three Keras/TensorFlow notebooks covering computer vision, NLP classification and sequence-to-sequence translation.

| Notebook | Task | Model | Result (validation) |
| :--- | :--- | :--- | :--- |
| [`cifar10_image_classification_cnn.ipynb`](notebooks/cifar10_image_classification_cnn.ipynb) | Classify 32×32 colour images into 10 classes ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)) | CNN with batch normalisation and dropout, trained with `EarlyStopping`, `ReduceLROnPlateau` and `ModelCheckpoint` callbacks; three milestone models compared | ~55% accuracy |
| [`tweet_sentiment_analysis.ipynb`](notebooks/tweet_sentiment_analysis.ipynb) | Positive / negative sentiment on tweets (Sentiment140, 10% sample) | `TextVectorization` + frozen [GloVe](https://nlp.stanford.edu/projects/glove/) 100-d embeddings + bidirectional LSTM | ~75% accuracy |
| [`english_chinese_translation_transformer.ipynb`](notebooks/english_chinese_translation_transformer.ipynb) | Translate English ↔ Chinese ([ManyThings](http://www.manythings.org/anki/) sentence pairs) | Transformer encoder–decoder with `MultiHeadAttention`, masked loss/accuracy | ~65% masked token accuracy |

## What each notebook shows

- **CIFAR-10 CNN** – data exploration, normalisation, Keras callbacks and iterative model building. The training/validation gap is documented in the notebook, which makes it a good case study in overfitting and regularisation.
- **Tweet sentiment** – parsing raw CSV, train/validation split, tokenisation with truncation and padding, vocabulary size tuning and using pre-trained word embeddings.
- **English–Chinese translation** – text standardisation for both languages, vocabulary building and index-to-word lookup, positional embeddings, attention-based encoder–decoder and greedy decoding.

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

The notebooks download their own data (CIFAR-10 via `keras.datasets`, GloVe and the translation pairs via URL). The tweet notebook expects `training_cleaned.csv` (the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset) in the working directory. A GPU (e.g. Google Colab) is recommended for the CNN and Transformer.
