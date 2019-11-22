import json

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from rnn_text_classifier import RNNTextClassifier


def read_and_preprocess():
    df = pd.read_csv("data/consumer_complaints_small.csv")
    df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].astype(str)

    print(df.head())

    df.info()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=configs["vocabulary_size"],
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
        lower=True)

    print(df['consumer_complaint_narrative'].values)

    tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['consumer_complaint_narrative'].values)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=configs["sequence_size"])
    print('Shape of data tensor:', X.shape)

    Y = pd.get_dummies(df['product']).values
    print('Shape of label tensor:', Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    # get the sample configs file
    with open("sample_configs.json") as config_file:
        configs = json.load(config_file)

    # train and test split
    X_train, X_test, Y_train, Y_test = read_and_preprocess()

    # rnn text classifier model
    model = RNNTextClassifier(configs)

    # training the model
    model.fit(X_train, Y_train)

    # testing accuracy
