from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.random.randint(low=1, high=50, size=(10, 2))
y = np.random.randint(low=0, high=2, size=(10, 4))

print(X)
print(y)

clf = LogisticRegression()

clf.fit(X, y)

print('=========  predictions =========')
print(clf.predict(X))

print('========= prob predictions =========')
print(np.round(clf.predict_proba(X), decimals=2))