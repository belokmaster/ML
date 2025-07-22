import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([(3, 2), (4, 3), (7, 5), (6, 4)])
y_train = np.array([0, 0, 1, 1])

new_animal = np.array([(5, 2.5)])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

answer_class = knn.predict(new_animal)
if answer_class[0] == 0:
    print("Новое животное - это кот")
else:
    print("Новое животное - это собака")