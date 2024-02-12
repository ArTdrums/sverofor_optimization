from keras.layers import Dense  # это класс полносвязной нейросети
import tensorflow as tf
from keras.models import Sequential  # это класс последовательно слоев - можно создавать
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.activations import linear, sigmoid

# уходим от случайной генераци весов
# сколько угодно слоев
tf.random.set_seed(7)
model = Sequential([
    Dense(12, input_shape=(1,), activation='linear'),
    Dense(6, input_shape=(1,), activation='linear'),
    Dense(3, input_shape=(1,), activation='linear'),
    Dense(1, input_shape=(1,), activation='linear')

])
# input_shape -входная разменость обьекта, activation- функция активации, layers -слой
# model.summary() # получение резюме модели

# w1, w0 = model.get_weights()

# придумываем архитектуру сети
# дата фрейм для вертикального движения
X_vpered_1 = np.array([25, 20, 4, 1, 18, 19, 18, 19, 2, 2, 2, 2]) / 25.0 * 0.99 + 0.01
X_vpered_1_2 = np.array([25, 18, 5, 2, 19, 20, 19, 20, 3, 3, 3, 3]) / 25.0 * 0.99 + 0.01
X_vpered_1_3 = np.array([25, 21, 6, 3, 20, 20, 19, 20, 4, 3, 4, 3]) / 25.0 * 0.99 + 0.01
X_vpered_1_4 = np.array([25, 19, 2, 4, 20, 21, 20, 17, 1, 2, 3, 4]) / 25.0 * 0.99 + 0.01

# выходожные значения вертикального движения
y_vpered_1 = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])

X_vpered_2 = np.array([7, 8, 7, 35, 30, 4, 3, 4, 23, 23, 25, 23]) / 35.0 * 0.99 + 0.01
X_vpered_2_2 = np.array([6, 6, 8, 34, 29, 2, 4, 2, 22, 22, 23, 26]) / 35.0 * 0.99 + 0.01
X_vpered_2_3 = np.array([5, 5, 6, 33, 28, 3, 2, 3, 25, 20, 22, 20]) / 35.0 * 0.99 + 0.01
X_vpered_2_4 = np.array([7, 8, 7, 35, 30, 4, 3, 4, 23, 23, 25, 23]) / 35.0 * 0.99 + 0.01

# выходожные значения для горизонтального движения
y_vpered_2 = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])

X_povorot_1 = np.array([25, 3, 3, 2, 1, 3, 22, 22, 21, 21, 3, 3]) / 25.0 * 0.99 + 0.01
X_povorot_2 = np.array([22, 5, 5, 3, 4, 3, 25, 21, 18, 21, 1, 3]) / 25.0 * 0.99 + 0.01
X_povorot_3 = np.array([21, 2, 3, 5, 3, 4, 19, 25, 23, 25, 2, 6]) / 25.0 * 0.99 + 0.01
X_povorot_4 = np.array([18, 2, 3, 2, 1, 3, 18, 17, 21, 24, 1, 1]) / 25.0 * 0.99 + 0.01

y_povorot_1 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])

# дата фрейм отдельное включение пешеходных переходов
X_pesh_1 = np.array([5, 6, 7, 8, 18, 21, 17, 19, 19, 22, 21, 18]) / 22.0 * 0.99 + 0.01
X_pesh_2 = np.array([6, 5, 8, 7, 22, 18, 22, 21, 17, 19, 19, 22]) / 22.0 * 0.99 + 0.01
X_pesh_3 = np.array([3, 8, 5, 4, 17, 16, 19, 22, 22, 17, 18, 15]) / 22.0 * 0.99 + 0.01
X_pesh_4 = np.array([3, 4, 2, 5, 22, 17, 12, 16, 19, 15, 22, 22]) / 22.0 * 0.99 + 0.01

y_pesh = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# математическое представление работы весов с сигналами
'''res = w1 *X[:1] +w0
res_2 = linear(res)'''
data_drame = [X_pesh_1, X_pesh_2, X_pesh_3, X_pesh_4]
model.compile(optimizer='sgd', loss='mse', metrics='mae')
for i in data_drame:
    print(model.fit(i, y_pesh, epochs=200))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_pesh),
        'пердсказания': np.squeeze(model.predict(i)),
        'пердсказания_2': np.squeeze(model.predict(X_pesh_2)),
    }))

# тестирование нейронной сети
check_1, check_2 = 5, -3

# w2, w3 = model.get_weights()
''' округление доцелого числа если надо
b = np.array([int(i) for i in np.squeeze(model.predict(X))])

print('было ', w1, w0, '\n', 'стало', w2, w3)
'''

print(model.predict(np.array([6, 5, 8, 7, 22, 18, 22, 21, 17, 19, 19, 22]) / 25.0 * 0.99 + 0.01))
