import json

import tensorflow as tf


class RNNTextClassifier:

    def __init__(self, configs):
        self.configs = configs

        # model definition
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.configs["vocabulary_size"], self.configs["embedding_size"]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(self.configs["number_of_classes"], activation='softmax'),

        ])

        #
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=['accuracy'])

    def fit(self, X, Y):
        '''
        Make the prediction
        '''

        return self.model.fit(X,
                              Y,
                              epochs=self.configs["epochs"],
                              batch_size=self.configs["batch_size"],
                              validation_split=self.configs["validation_split"],
                              callbacks=self.get_callbacks()
                              )

    def predict(self, X):
        '''
        Make the prediction
        '''
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=self.configs["sequence_size"]
        )

        return self.model.predict(padded)

    def get_callbacks(self):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.configs["checkpoint_path"],
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True, save_weights_only=False,
                save_frequency=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=0.0001
            )
        ]

    @staticmethod
    def get_configs():
        '''
        Make the prediction
        '''
        with open("sample_configs.json") as config_file:
            configs = json.load(config_file)
        return configs
