import numpy as np

X_train = np.array([(3, 2), (4, 3), (7, 5), (6, 4)])
y_train = np.array([0, 0, 1, 1])

new_animal = np.array([5, 2.5])

distance = []
for i in range(len(X_train)):
    dist = np.sqrt((X_train[i][0] - new_animal[0])**2 + (X_train[i][1] - new_animal[1])**2)
    distance.append(dist)

nearest_indices = np.argsort(distance)[:3]
nearest_labels = y_train[nearest_indices]

answer_class = np.bincount(nearest_labels).argmax()
if answer_class == 0:
    print("Новое животное - это кот")
else:
    print("Новое животное - это собака")