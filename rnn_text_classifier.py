import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class RNNTextClassifier(tf.keras.Model, BaseEstimator, ClassifierMixin):
    """ Defines a Long Short Term Memory (LSTM) based model as an approach in the text classification problem.
    """

    def __init__(self, configs, embedding_matrix):
        super(RNNTextClassifier, self).__init__()

        self.configs = configs

        # model definition
        self.model = tf.keras.Sequential([
            # embedding layer
            tf.keras.layers.Embedding(self.configs["vocabulary_size"],
                                      self.configs["embedding_size"],
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=False
                                      ),
            # dropout layer
            tf.keras.layers.Dropout(0.2),
            # lstm layer
            tf.keras.layers.LSTM(self.configs["hidden_size"], dropout=0.2, recurrent_dropout=0.2),
            # dense layer
            tf.keras.layers.Dense(self.configs["number_of_classes"], activation='softmax'),

        ])

        # configures the model for training
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(self.configs["learning_rate"]),
                           metrics=['accuracy'])

    def fit(self, train_texts, train_scores, val_texts, val_scores, tokenizer):
        """
        Trains the RNNTextClassifier model for a maximum number of epochs.
        :param train_texts: list of training texts.
        :param train_scores: list of scores (classes) for each training text.
        :param val_texts: list of validation texts,
        :param val_scores: list of scores (classes) for each validation text.
        :param tokenizer: text tokenization utility class.
        :return: the training history.
        """

        x_train, y_train = self.preprocess(train_texts, train_scores, tokenizer)
        x_val, y_val = self.preprocess(val_texts, val_scores, tokenizer)

        return self.model.fit(x=x_train,
                              y=y_train,
                              epochs=self.configs["epochs"],
                              batch_size=self.configs["batch_size"],
                              validation_data=(x_val, y_val),
                              callbacks=[tf.keras.callbacks.EarlyStopping(
                                  monitor='val_loss',
                                  patience=3,
                                  min_delta=0.0001)
                              ]
                              )

    def predict(self, texts, tokenizer):
        """
        :param texts: a list of texts.
        :return: predicted tensor.
        """

        tokenized_texts = tokenizer.texts_to_sequences(texts)

        padded_texts = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_texts,
            maxlen=self.configs["sequence_size"]
        )

        return self.model.predict(padded_texts)

    def preprocess(self, texts, scores, tokenizer):
        """
        Vectorize the text samples into a 2D integer tensor
        :param texts: a list of texts.
        :param scores: list of scores (classes) for each text.
        :param tokenizer: text tokenization utility class
        :return: 2D integer tensor.
        """
        x = None
        y = None

        if not (texts is None or scores is None):
            x = tokenizer.texts_to_sequences(texts)  # transforms each text in texts to a sequence of integers.
            x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.configs["sequence_size"])  # padding
            y = pd.get_dummies(scores).values  # scores representation

        return x, y
