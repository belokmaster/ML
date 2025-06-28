import numpy as np
import matplotlib.pyplot as plt

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

def create_features(x):
    return np.array([np.ones_like(x), x, x**2, x**3]).T


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 50 # размер мини-батча (величина K = 50)

Qe = 0.0
Qe_history = []
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
    k = np.random.randint(0, sz - batch_size)
    batch_x = coord_x[k:k+batch_size]
    batch_y = coord_y[k:k+batch_size]

    X_batch = create_features(batch_x)
    
    predictions = X_batch @ w
    
    errors = predictions - batch_y
    Qk = np.mean(errors**2)
    
    Qe = lm * Qk + (1 - lm) * Qe
    Qe_history.append(Qe)
    
    gradient = (2 / batch_size) * X_batch.T @ errors
    
    w = w - eta * gradient

X_all = create_features(coord_x)
predictions = X_all @ w
Q_final = np.mean((predictions - coord_y)**2)
Q = np.mean((predictions - coord_y)**2)

X_all = create_features(coord_x)
predictions = X_all @ w

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(Qe_history, label='Qe (эксп. скользящее среднее)', color='orange')
plt.xlabel('Итерация')
plt.ylabel('Значение Qe')
plt.title('Изменение Qe в процессе обучения')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(coord_x, coord_y, label='Исходная функция')
plt.plot(coord_x, predictions, '--', label='Аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение исходной функции и аппроксимации')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Веса модели: w0 = {w[0]:.4f}, w1 = {w[1]:.4f}, w2 = {w[2]:.4f}, w3 = {w[3]:.4f}")
print(f"Финальный Q(a,X) = {Q_final:.4f}")