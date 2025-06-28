import numpy as np
import matplotlib.pyplot as plt

# Исходная функция
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

# Генерация данных
coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)
sz = len(coord_x)

# Параметры SG
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
w = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02

# Для хранения истории Qe
Qe_history = []
Qe = 0.0
np.random.seed(0)

# Обучение модели
for i in range(N):
    k = np.random.randint(0, sz)
    x = coord_x[k]
    y = coord_y[k]
    
    x_k = np.array([1, x, x**2, np.cos(2*x), np.sin(2*x)]) # вектор признаков
    a_x = np.dot(w, x_k) # предсказание модели

    L_k = (a_x - y)**2 # функция потерь

    Qe = lm * L_k + (1 - lm) * Qe
    Qe_history.append(Qe)
    
    gradient = 2 * (a_x - y) * x_k
    w = w - eta * gradient

# Вычисление итогового Q
X = np.array([np.ones_like(coord_x), coord_x, coord_x**2, 
              np.cos(2*coord_x), np.sin(2*coord_x)]).T

predictions = np.dot(X, w)

Q = np.mean((predictions - coord_y)**2)

plt.figure(figsize=(12, 5))

# График изменения Qe
plt.subplot(1, 2, 1)
plt.plot(Qe_history, label='Qe (эксп. скользящее среднее)', color='orange')
plt.xlabel('Итерация')
plt.ylabel('Значение Qe')
plt.title('Изменение Qe в процессе обучения')
plt.grid(True)
plt.legend()

# График аппроксимации
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

print(f"Итоговые веса модели: {w}")
print(f"Итоговый средний эмпирический риск Q: {Q:.4f}")
print(f"Последнее значение Qe: {Qe:.4f}")