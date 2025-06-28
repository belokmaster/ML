import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """Исходная функция, которую необходимо аппроксимировать"""
    return 0.1 * x**2 - np.sin(x) + 5.

def s(x):
    """Создает матрицу признаков S по вектору x."""
    return np.array([np.ones_like(x), x, x**2, x**3]).T

def a(S_matrix, w):
    """Вычисляет предсказания модели по готовой матрице S и весам w."""
    return S_matrix @ w

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)    # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма

# Модель: a(x) = w0*1 + w1*x + w2*x^2 + w3*x^3
# вектор признаков s = [1, x, x^2, x^3].
S_matrix = s(coord_x)

for i in range(N):
    predictions = a(S_matrix, w)
    errors = predictions - coord_y
    gradient = (2 / sz) * (S_matrix.T @ errors)
    w = w - eta * gradient

final_predictions = a(S_matrix, w)
Q = np.mean((final_predictions - coord_y)**2)
w = list(w)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(coord_x, coord_y, label='Исходная функция', linewidth=2)
plt.plot(coord_x, final_predictions, '--', label='Аппроксимация', linewidth=2)
plt.title('Аппроксимация функции кубическим полиномом')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print(f"Полученные веса: w0 = {w[0]:.4f}, w1 = {w[1]:.4f}, w2 = {w[2]:.4f}, w3 = {w[3]:.4f}")
print(f"Среднеквадратичная ошибка: Q = {Q:.6f}")