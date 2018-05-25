from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from run_multilabel_classifier import run


def keras_logreg_model():
	model = Sequential()
	model.add(Dense(units=1, input_shape=(300,), kernel_initializer='normal', activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


param_grid = {
        'estimator__bag_of_words__stop_words': ['english'],
        'estimator__bag_of_words__ngram_range': [(1, 2)],
        'estimator__bag_of_words__max_features': [500],
        'estimator__dim_reduct__n_components': [300],
        'estimator__normalizer__norm': ['l2']
       # 'estimator__classifier__C': [100., 0.1, 0.0001]
}

estimator = KerasClassifier(build_fn=keras_logreg_model, epochs=1, batch_size=16, verbose=1)

run(param_grid, estimator)
