import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

X_train = np.c_[0.5, 1].T
y_train = [0.5, 1]
X_test = np.c_[0, 2].T

np.random.seed(0)

classifiers = dict(
    ols=linear_model.LinearRegression(), 
    ridge=linear_model.Ridge(alpha=0.1)
)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, clf) in zip(axs, classifiers.items()):
    slopes = []  

    for _ in range(10):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X_train
        clf.fit(this_X, y_train)

        slope = clf.coef_[0]
        slopes.append(slope)

        ax.plot(X_test, clf.predict(X_test), color="gray", alpha=0.7)
        ax.scatter(this_X, y_train, s=3, c="gray", marker="o", zorder=10)

    clf.fit(X_train, y_train)
    ax.plot(X_test, clf.predict(X_test), linewidth=2, color="blue")
    ax.scatter(X_train, y_train, s=30, c="red", marker="+", zorder=10)

    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim((0, 1.6))
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.grid(True)

    slopes = np.array(slopes)
    print(f"{name.upper()} slope variance: {np.var(slopes):.6f}")

fig.tight_layout()
plt.show()