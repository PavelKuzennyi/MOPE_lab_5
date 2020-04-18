import numpy as np
from prettytable import PrettyTable
from scipy.stats import f, t
from functools import partial
import sklearn.linear_model as lm

# Variant №208

x1_min = -5
x1_max = 6
x2_min = -7
x2_max = 9
x3_min = -5
x3_max = 3

m = 3
N = 15

x_max_av = (x1_max + x2_max + x3_max) / 3
x_min_av = (x1_min + x2_min + x3_min) / 3

y_max = int(200 + x_max_av)
y_min = int(200 + x_min_av)

x_matrix = np.array([
    [x1_min, x2_min, x3_min],
    [x1_min, x2_max, x3_max],
    [x1_max, x2_min, x3_max],
    [x1_max, x2_max, x3_min]
])

x1_0 = (x1_min+x1_max)/2
x2_0 = (x2_min+x2_max)/2
x3_0 = (x3_min+x3_max)/2


while True:
    while True:
        matrix_plan = np.random.randint(y_min, y_max, size=(N, m))

        x0_factor = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        x1_factor = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1.215, 1.215, 0, 0, 0, 0, 0])
        x2_factor = np.array([-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -1.215, 1.215, 0, 0, 0])
        x3_factor = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1.215, 1.215, 0])
        x1x2_factor = x1_factor * x2_factor
        x1x3_factor = x1_factor * x3_factor
        x2x3_factor = x2_factor * x3_factor
        x1x2x3_factors = x1_factor * x2_factor * x3_factor
        x_1_2_factor = x1_factor * x1_factor
        x_2_2_factor = x2_factor * x2_factor
        x_3_2_factor = x3_factor * x3_factor

        x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        x1 = np.array([x1_min, x1_min, x1_min, x1_min, x1_max, x1_max, x1_max, x1_max, x1_min * 1.215, round(x1_max * 1.215), x1_0, x1_0, x1_0, x1_0, x1_0])
        x2 = np.array([x2_min, x2_min, x2_max, x2_max, x2_min, x2_min, x2_max, x2_max, x2_0, x2_0, x2_min * 1.215, x2_max * 1.215, x2_0, x2_0, x2_0])
        x3 = np.array([x3_min, x3_max, x3_min, x3_max, x3_min, x3_max, x3_min, x3_max, x3_0, x3_0, x3_0, x3_0, x3_min * 1.215, round(x3_max * 1.215), x3_0])
        x1x2 = x1 * x2
        x1x3 = x1 * x3
        x2x3 = x2 * x3
        x1x2x3 = x1 * x2 * x3
        x_1_2 = x1 * x1
        x_2_2 = x2 * x2
        x_3_2 = x3 * x3

        factor_matrix = np.zeros((N, 11))
        factor_matrix[:,0] = x0_factor
        factor_matrix[:,1] = x1_factor
        factor_matrix[:, 2] = x2_factor
        factor_matrix[:, 3] = x3_factor
        factor_matrix[:, 4] = x1x2_factor
        factor_matrix[:, 5] = x1x3_factor
        factor_matrix[:, 6] = x2x3_factor
        factor_matrix[:, 7] = x1x2x3_factors
        factor_matrix[:, 8] = x1 * x1
        factor_matrix[:, 9] = x2 * x2
        factor_matrix[:, 10] = x3 * x3

        x_norm = np.zeros((N, 11))

        x_norm[:, 0] = x0
        x_norm[:, 1] = x1
        x_norm[:, 2] = x2
        x_norm[:, 3] = x3
        x_norm[:, 4] = x1x2
        x_norm[:, 5] = x1x3
        x_norm[:, 6] = x2x3
        x_norm[:, 7] = x1x2x3
        x_norm[:, 8] = x1x2x3
        x_norm[:, 9] = x1x2x3
        x_norm[:, 10] = x1x2x3

        y_average = np.zeros((N, 1))
        for i in range(N):
            y_average[i, 0] = round((sum(matrix_plan[i, :] / m)), 3)

        d_list = np.zeros((N, 1))
        np.array(d_list)
        for i in range(N):
            d_list[i][0] = (
                round(((matrix_plan[i][0] - y_average[i][0]) ** 2 + (matrix_plan[i][1] - y_average[i][0]) ** 2 + (
                        matrix_plan[i][2] - y_average[i][0]) ** 2) / 3, 3))
        d_sum = sum(d_list)

        my_table = np.hstack((x_norm, matrix_plan, y_average, d_list))

        table = PrettyTable()
        table.field_names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "X1^2", "X2^2", "X3^2", "Y1", "Y2", "Y3",
                             "Y", "S^2"]
        for i in range(len(my_table)):
            table.add_row(my_table[i])

        print(f'\nМатриця планування при N = {N}, m = {m}')

        print(table)
        y_average = y_average[:, 0]
        skm = lm.LinearRegression(fit_intercept=False)  # знаходимо коефіцієнти рівняння регресії
        skm.fit(x_norm, y_average)
        B = skm.coef_

        print('\nРівняння регресії:')

        print("\ny = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 + {}*x1^2 + {}*x2^2 + {}*x3^2 \n".format(
            round(float(B[0]), 3),
            round(float(B[1]), 3),
            round(float(B[2]), 3),
            round(float(B[3]), 3),
            round(float(B[4]), 3),
            round(float(B[5]), 3),
            round(float(B[6]), 3),
            round(float(B[7]), 3),
            round(float(B[8]), 3),
            round(float(B[9]), 3),
            round(float(B[10]), 3)))

        print('\nКоефіцієнти рівняння регресії:')
        B = [round(i, 3) for i in B]
        print(B, "\n")
        y_list = np.zeros((15, 2))
        y_list[:, 0] = np.dot(x_norm, B)
        y_list[:, 1] = y_average
        my_table = y_list
        table = PrettyTable()
        table.field_names = ["Фактори", "Y середнє"]
        for i in range(len(my_table)):
            table.add_row(my_table[i])
        print(table)

        print('\nПроведемо перевірку рівняння')

        f1 = m - 1
        f2 = N
        f3 = f1 * f2
        q = 0.05

        student = partial(t.ppf, q=1 - q)
        t_student = student(df=f3)
        q1 = q / f1
        fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
        G_kr = fisher_value / (fisher_value + f1 - 1)

        y_aver = [round(sum(i) / len(i), 3) for i in matrix_plan]
        print('\nСереднє значення y:', y_aver)

        res = []
        for i in range(N):
            s = sum([(y_aver[i] - matrix_plan[i][j]) ** 2 for j in range(m)]) / m
            res.append(round(s, 3))
        disp = res
        print('Дисперсія y:', disp)

        S_kv = res
        Gp = max(S_kv) / sum(S_kv)
        print('\nПеревірка за критерієм Кохрена')

        print(f'Gp = {Gp}')
        if Gp < G_kr:
            print(f'З ймовірністю {1-q} дисперсії однорідні.')
            break
        else:
            print("Необхідно збільшити кількість дослідів")
            m += 1


    X = factor_matrix[:, 1:]
    s_kv_aver = sum(S_kv) / N


    s_Bs = (s_kv_aver / N / m) ** 0.5
    res = [sum(1 * y for y in y_aver) / N]

    for i in range(len(X[0])):
        b = sum(j[0] * j[1] for j in zip(X[:, i], y_aver)) / N
        res.append(b)
    Bs = res
    ts = [round(abs(B) / s_Bs, 3) for B in Bs]

    print('\nКритерій Стьюдента:\n', ts)
    res = [t for t in ts if t > t_student]
    final_k = [B[i] for i in range(len(ts)) if ts[i] in res]
    print('\nКоефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
        [i for i in B if i not in final_k]))

    y_new = []
    for j in range(N):
        x = [x_norm[j][i] for i in range(len(ts)) if ts[i] in res]
        b = final_k
        y_new.append(sum([x[i] * b[i] for i in range(len(x))]))
    print(f'\nЗначення "y" з коефіцієнтами {final_k}')
    d = len(res)
    print("Кількість значимих коефіцієнтів d =", d)
    if d >= N:
        print('\nF4 <= 0')
        print('')
    f4 = N - d

    S_ad = m / (N - d) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(matrix_plan))])
    S_kv_aver = sum(S_kv) / N
    F_p = S_ad / S_kv_aver

    fisher = partial(f.ppf, q=0.95)
    f_t = fisher(dfn=f4, dfd=f3)
    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', F_p)
    print('Ft =', f_t)
    if F_p < f_t:
        print('Отримана математична модель адекватна експериментальним даним')
        break
    else:
        print('Рівняння регресії неадекватно оригіналу')
        print('\nПочинаємо з початку\n')
        print("----------------------------------------")
