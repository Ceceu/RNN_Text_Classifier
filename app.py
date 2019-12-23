import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from rnn_text_classifier import RNNTextClassifier


def get_splits(texts, scores):
    # example splits
    train_texts, test_texts, train_scores, test_scores = train_test_split(
        texts, scores, test_size=0.10, random_state=42
    )
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        train_texts, train_scores, test_size=0.10, random_state=42
    )

    return train_texts, train_scores, test_texts, test_scores, val_texts, val_scores


def preprocessing_dataset(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=configs["vocabulary_size"])
    tokenizer.fit_on_texts(texts)
    return tokenizer


def get_word_embeddings(embeddings_path):
    word_embeddings = {}
    with open(embeddings_path, "r") as embeddings_file:
        for line in embeddings_file:
            word, embedding = line.split(maxsplit=1)
            embedding = np.fromstring(embedding, 'f', sep=' ')
            word_embeddings[word] = embedding
    return word_embeddings


def get_embedding_matrix(word_index, embeddings_path):
    embeddings = get_word_embeddings(embeddings_path)
    embedding_matrix = np.zeros((configs["vocabulary_size"], configs["embedding_size"]))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None and i < configs["vocabulary_size"]:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_texts_and_scores(dataset_path):
    with open(dataset_path + "texts.txt") as texts_file:
        texts = texts_file.readlines()
    with open(dataset_path + "scores.txt") as scores_file:
        scores = scores_file.readlines()

    return texts, scores


if __name__ == '__main__':

    # get the sample configs file
    with open("sample_configs.json") as config_file:
        configs = json.load(config_file)

    # getting list of texts and scores
    print("Reading dataset.")
    texts, scores = get_texts_and_scores(dataset_path="data/datasets/acm/")

    # creating vocabulary
    print(f"Creating vocabulary with the {configs['vocabulary_size']} most frequent words.")
    tokenizer = preprocessing_dataset(texts)

    # prepare embedding matrix
    print("Getting word embeddings.")
    embedding_matrix = get_embedding_matrix(
        tokenizer.word_index,
        "data/embeddings/glove.6B.100d.txt")

    # train and test split
    print("Getting train, test and validation splits.")
    train_texts, train_scores, test_texts, test_scores, val_texts, val_scores, = get_splits(
        texts,
        scores
    )

    # rnn text classifier model
    model = RNNTextClassifier(configs, embedding_matrix)

    # training the model
    print("Trainng model.")
    model.fit(train_texts, train_scores, val_texts, val_scores, tokenizer)

    # predict
    print("Predictin with sample text: modeling and analyzing java software architectures")
    new_text = ["modeling and analyzing java software architectures."]

    predictions = model.predict(new_text, tokenizer)

    labels = list(set(scores))
    print(predictions, labels[np.argmax(predictions)])
