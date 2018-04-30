from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from run_binary_classifier import run


def keras_logreg_model():
	model = Sequential()
	model.add(Dense(units=1, input_shape=(2,), kernel_initializer='normal', activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


param_grid = {
        'bag_of_words__stop_words': ['english'],
        'bag_of_words__ngram_range': [(1, 2)],
        'bag_of_words__max_features': [500],
        'dim_reduct__n_components': [300]
        #'normalizer__norm': ['l2']
        #'classifier__C': [100., 0.1, 0.0001]
}

estimator = KerasClassifier(build_fn=keras_logreg_model, epochs=1, batch_size=5, verbose=1)

run(param_grid, estimator)
