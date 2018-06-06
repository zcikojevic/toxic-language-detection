from keras.layers import (LSTM, Activation, Dense, Dropout,
                          Embedding, GlobalMaxPool1D, Input)
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from run_multilabel_classifier import run

def create_model():
    inp = Input(shape=())
    embed_size = 128
    x = Embedding(2000, embed_size)(inp)
    x = LSTM(60, return_sequences=True,name='lstm_layer')(inp)
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

param_grid = {
    'estimator__classifier__C': [100., 0.1]
}

estimator = KerasClassifier(build_fn=create_model, epochs=1, batch_size=16, verbose=1)

run(param_grid, estimator)