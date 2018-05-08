from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.layers import (LSTM, Activation, Dense, Dropout,
                          Embedding, GlobalMaxPool1D, Input)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from run_binary_classifier import _load_comments, run


def create_model(max_features):
    inp = Input(shape=(200, ))
    embed_size = 128
    x = Embedding(max_features, embed_size)(inp)
    x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


comments_X, comments_y = _load_comments('../../../data/train_binary_labels.csv')
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(comments_X))
comments_X = tokenizer.texts_to_sequences(comments_X)
comments_X = pad_sequences(comments_X, maxlen=200)

comments_X_train, comments_X_test, comments_y_train, comments_y_test = train_test_split(comments_X, comments_y, train_size=0.7, random_state=1)

model = create_model(comments_X_train.shape[0])

hist = model.fit(comments_X_train, comments_y_train, batch_size=64, epochs=1, validation_split=0.2)

model.save('keras_lstm_binary.h5')

print('\n=================  Classification report  =================')

predictions = model.predict(comments_X_test, batch_size=64)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

print(classification_report(comments_y_test, predictions))
