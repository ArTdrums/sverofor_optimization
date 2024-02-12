Для начала пронумеруем светофоры для присвоения им id.
Составляем мат модель для решения задачи, для этого нам потребуется:
# получение данных (фиксация трафика) на каждый светофор.
# разборка алгоритма включения светофором в зависимости от трафика
# оптимизация с помощью нейронной сети


Получение входных данных для светофоров.
Т.к данные взять неоткуда по условию задачи, я сгенерировал их самостоятельно (по своему усмотрению)

Алгоритм включения светофоров.
Я выделил несколько основных движений:
# по вертикали сверху вниз (1, 2, 5, 6, 7, 8)
# по горизонтали снизу-вверх (3, 4, 10, 9, 11, 12)
# поворот на права сверху вниз (1, 10, 9, 7, 8)
# поворот на права снизу-вверх (2, 5, 6, 11, 12)
# для пешеходов (5, 6, 7, 8, 9, 10, 11, 12)

Нейронная сеть написана 2 способами
1. С помощью фреймворка tensorflow.
2. Рукописная нейросеть.


 

