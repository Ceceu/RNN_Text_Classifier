import tensorflow as tf


class RNNTextClassifier(tf.keras.Model):
    """ Defines a Long Short Term Memory (LSTM) based model as an approach in the text classification problem.
    """

    def __init__(self, configs):
        super(RNNTextClassifier, self).__init__()

        self.configs = configs

        # model definition
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.configs["vocabulary_size"], self.configs["embedding_size"]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(self.configs["number_of_classes"], activation='softmax'),

        ])

        # configures the model for training
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(self.configs["learning_rate"]),
                           metrics=['accuracy'])

    def fit(self, X, Y):
        """ Trains the RNNTextClassifier model for a maximum number of epochs.
        :param X: input tensor with shape (m:samples, n: max text length).
        :param Y: target tensor with shape (m:samples, n: number of classes).
        :return: the training history.
        """
        return self.model.fit(X,
                              Y,
                              epochs=self.configs["epochs"],
                              batch_size=self.configs["batch_size"],
                              validation_split=self.configs["validation_split"],
                              callbacks=self.get_callbacks()
                              )

    def predict(self, X):
        """
        :param X: input tensor with shape (m:samples, n: max text length).
        :return: predicted tensor with shape (m:samples, n: number of classes).
        """

        padded = tf.keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=self.configs["sequence_size"]
        )

        return self.model.predict(padded)

    def get_callbacks(self):
        """
        :return:
        """
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.configs["checkpoint_path"],
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                save_frequency=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=0.0001
            )
        ]
