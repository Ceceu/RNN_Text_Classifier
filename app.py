import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from rnn_text_classifier import RNNTextClassifier


def read_and_preprocess():
    # read data
    df = pd.read_csv("data/consumer_complaints_small.csv")
    df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].astype(str)

    # data information
    print("Data:\n", df.head())

    print("Text classes = ", set(df["product"].values))

    tokenizer = get_simple_tokenizer()

    tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)
    word_index = tokenizer.word_index

    X = tokenizer.texts_to_sequences(df['consumer_complaint_narrative'].values)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=configs["sequence_size"])

    Y = pd.get_dummies(df['product']).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    print("Train shape:\n", X_train.shape, Y_train.shape)
    print("Test shape:\n", X_test.shape, Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def train(configs, X_train, Y_train):
    # rnn text classifier model
    model = RNNTextClassifier(configs)

    # training the model
    model.fit(X_train, Y_train)


def test(configs, X_test, Y_test):
    model = tf.keras.models.load_model(configs["checkpoint_path"])
    loss, acc = model.evaluate(X_test, Y_test)
    print('Loss: {:0.3f} - Accuracy: {:0.3f}'.format(loss, acc))


def predict(new_text, configs):
    tokenizer = get_simple_tokenizer()
    tokenized_text = tokenizer.texts_to_sequences(new_text)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_text,
        maxlen=configs["sequence_size"]
    )
    model = model = tf.keras.models.load_model(configs["checkpoint_path"])
    predictions = model.predict(padded_text)

    labels = ['Mortgage', 'Credit reporting', 'Money transfers', 'Credit card',
              'Prepaid card', 'Payday loan', 'Debt collection', 'Consumer Loan',
              'Bank account or service', 'Other financial service', 'Student loan']
    print(predictions, labels[np.argmax(predictions)])


def get_simple_tokenizer():
    return tf.keras.preprocessing.text.Tokenizer(
        num_words=configs["vocabulary_size"],
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
        lower=True)


if __name__ == '__main__':
    # get the sample configs file
    with open("sample_configs.json") as config_file:
        configs = json.load(config_file)

    # train and test split
    X_train, X_test, Y_train, Y_test = read_and_preprocess()

    # train model
    train(configs, X_train, Y_train)

    # testing accuracy
    test(configs, X_test, Y_test)

    # predict
    new_text = [
        "I am a victim of identity theft and someone stole my identity and personal information to open up a Visa "
        "credit card account with Bank of America. The following Bank of America Visa credit card account do not "
        "belong to me : XXXX."]

    predict(new_text, configs)
