import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from rnn_text_classifier import RNNTextClassifier


def read_and_preprocess(dataset_name):
    """ read and preprocess some dataset
    :return: the dataset preprocessed train, test and validation splits
    """

    # read data
    df = pd.read_csv(dataset_name)
    df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].astype(str)

    # data information
    print("Data:\n", df.head())

    print("Text classes = ", set(df["product"].values))

    tokenizer = get_simple_tokenizer()

    tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)

    x = tokenizer.texts_to_sequences(df['consumer_complaint_narrative'].values)
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=configs["sequence_size"])

    y = pd.get_dummies(df['product']).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    print("Train shape:\n", x_train.shape, y_train.shape)
    print("Test shape:\n", x_test.shape, y_test.shape)
    print("Validation shape:\n", x_val.shape, y_val.shape)

    return x_train, y_train, x_test, y_test, x_val, y_val


def train(configs, x_train, y_train, x_val, y_val):
    # rnn text classifier model
    model = RNNTextClassifier(configs)

    # training the model
    model.fit(x_train, y_train, x_val, y_val)


def test(configs, x_test, y_test):
    model = tf.keras.models.load_model(configs["checkpoint_path"])
    loss, acc = model.evaluate(x_test, y_test)
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
    x_train, y_train, x_test, y_test, x_val, y_val = read_and_preprocess(
        dataset_name="data/consumer_complaints_small.csv"
    )

    # train model
    train(configs, x_train, y_train, x_val, y_val)

    # testing accuracy
    test(configs, x_test, y_test)

    # predict
    new_text = [
        "I am a victim of identity theft and someone stole my identity and personal information to open up a Visa "
        "credit card account with Bank of America. The following Bank of America Visa credit card account do not "
        "belong to me : XXXX."]

    predict(new_text, configs)
