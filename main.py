import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f
import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

path_x1 = './x1.txt'
path_x2 = './x2.txt'
path_x3 = './x3.txt'
path_x4 = './x4.txt'
path_x5 = './x5.txt'
path_y = './y.txt'

df = pd.DataFrame({
    "x1": np.loadtxt(path_x1, converters={0: np.int64}),
    "x2": np.loadtxt(path_x2, converters={0: np.int64}),
    # "x4": np.loadtxt(path_x4, converters={0: np.int64}),
    "x3": np.loadtxt(path_x3, converters={0: np.int64}),
    "x5": np.loadtxt(path_x5, converters={0: np.int64}),
    "y": np.loadtxt(path_y, converters={0: np.int64})
})

# Используйте столбец с метками времени в качестве индекса
df['timestamp'] = pd.period_range(start='2018-01', end='2022-12', freq='M')
df = df.set_index('timestamp')

data = df['y'].values
# Убираем умножение на 0.001 и округляем вверх
data = (data * 0.001).round().astype(int)


# Определение дискретного ряда распределения
def discrete_var(data):
    discrete = {}
    for line in data:
        discrete[line] = discrete.get(line, 0) + 1
    discrete = dict(sorted(discrete.items(), key=lambda item: item[0]))
    return discrete


def interval_var(data):
    # Число групп и длина интервалов
    discrete = discrete_var(data)
    m = int(np.ceil(1 + 3.222 * np.log10(max(data) - min(data))))
    h = int(np.ceil((max(data) - min(data)) / m))

    # Создание интервального ряда
    intervals = [(i, i + h) for i in range(min(data), max(data), h)]
    frequencies = [0] * len(intervals)

    # Распределение частот в интервалах
    for i, interval in enumerate(intervals):
        for key, value in discrete.items():
            if interval[0] <= key < interval[1]:
                frequencies[i] += value
    return intervals, frequencies, m, h


# Вывод интервального ряда распределения
print('Интервальное распределение', '\n')
intervals, frequencies, m, h = interval_var(data)
print("Интервал\tЧастота")
for i, interval in enumerate(intervals):
    print(f"[{interval[0]}, {interval[1]})\t{frequencies[i]}")

print()

# Построение гистограммы

def hist_xi(data, m):
    plt.hist(data, bins=m, range=(min(data), max(data)), edgecolor='black')
    plt.xlabel('Тыс. человек')
    plt.ylabel('Частота')
    plt.title('Гистограмма количества посещений')
    plt.show()


hist_xi(data, m)


# Нормальное распределение по теореме 3-х сигм
def normal_distribution_sigma(data):
    mean_x = (1 / len(data)) * sum(data)
    std_x = np.sqrt((sum([(i - mean_x) ** 2 for i in data]) / len(data)))

    sigma_68 = sum([1 if mean_x - std_x <= i <= mean_x + std_x else 0 for i in data]) / len(data) * 100
    sigma_95 = sum([1 if mean_x - 2 * std_x <= i <= mean_x + 2 * std_x else 0 for i in data]) / len(data) * 100
    sigma_99 = sum([1 if mean_x - 3 * std_x <= i <= mean_x + 3 * std_x else 0 for i in data]) / len(data) * 100

    return sigma_68, sigma_95, sigma_99


print('Нормальное распределение по теореме 3-х сигм')

# Правило 3-х сигм
sigma_68, sigma_95, sigma_99 = normal_distribution_sigma(data)
print(f"Процент вхождений в интервал 1 сигмы: {sigma_68}")
print(f"Процент вхождений в интервал 2 сигм: {sigma_95}")
print(f"Процент вхождений в интервал 3 сигм: {sigma_99}")

if (sigma_68 > 68) and (sigma_95 > 95) and (float(sigma_99) > 99.7):
    print('Распределение нормальное')
else:
    print('Распределение не нормальное')

print()

# Нормальное распределение по критерию Пирсона
print('Нормальное распределение по критерию Пирсона')

def normal_distribution_pearson(data):
    n = (sum(frequencies))
    xi = [(left + right) / 2 for left, right in intervals]

    def mean_xi(intervals, frequencies):
        return (1 / sum(frequencies)) * sum(
            [((left + right) / 2) * frequencies[i] for i, (left, right) in enumerate(intervals)])

    def var_xi(intervals, frequencies):
        return (1 / sum(frequencies)) * sum(
            [(((left + right) / 2) - mean_xi(intervals, frequencies)) ** 2 * frequencies[i] for i, (left, right) in
             enumerate(intervals)])

    def std_xi(intervals, hist_data):
        return np.sqrt(var_xi(intervals, hist_data))

    ui = [(i - mean_xi(intervals, frequencies)) / std_xi(intervals, frequencies) for i in xi]

    def laplace_xi(x):
        return np.exp(- (x ** 2 / 2)) / (np.sqrt(2 * np.pi))

    f_ui = [laplace_xi(i) for i in ui]
    ni_teor = [(h * n / std_xi(intervals, frequencies)) * i for i in f_ui]

    chi2_obs = sum([(frequencies[i] - j) ** 2 / j for i, j in enumerate(ni_teor)])

    r = 2
    chi2_crit = chi2.ppf(1 - 0.05, m - r - 1)
    return chi2_obs, chi2_crit, chi2_obs / chi2_crit


chi2_obs, chi2_crit, chi2_ratio = normal_distribution_pearson(data)
print(f'Хи-квадрат наблюдаемое {chi2_obs}')
print(f'Хи-квадрат критическое {chi2_crit}')
print(f'Рассчитанное значение {chi2_obs / chi2_crit}')
if chi2_obs > chi2_crit:
    print("Отвергаем H0, распределение не является нормальным")
else:
    print("Принимаем H0, распределение является нормальным")
print()


# Множественная регрессия
n, m = df.shape
m = m - 1


# Составим матрицу парных корреляций
def corr_analysis(df):
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", annot_kws={"size": 10})
    plt.show()
    print(f'Детерминант матрицы парных корреляций: {np.linalg.det(df.corr().to_numpy())}')

corr_matrix = df.corr()

print()
# Выведем матрицу корреляций
print("Матрица корреляций:")
print(corr_matrix)
corr_analysis(df)


# Составим матрицу признаков и вектор ответов
X = df.drop('y', axis=1)
y = df['y']

# Построение регрессионной модели с помощью statsmodels

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Добавление константы (интерцепта) к матрице признаков для МНК
X_train_mnk = sm.add_constant(X_train)
X_test_mnk = sm.add_constant(X_test)

# Реализация МНК для тренировочных данных
model = sm.OLS(y_train, X_train_mnk).fit()

# Получение прогнозов для тестовых данных
y_pred = model.predict(X_test_mnk)


# Тест Фишера
def fisher_test(y, X, model, n, m):
    # Общая сумма квадратов отклонений
    S2_y = np.sum((y - np.mean(y)) ** 2) / (n - 1)
    # Сумма квадратов отклонений объясненной моделью
    S2_fact = np.sum((model.predict(X) - np.mean(y)) ** 2) / m
    # Сумма квадратов отклонений по остаткам
    S2_e = np.sum((y - model.predict(X)) ** 2) / (n - m - 1)

    # F-статистика
    F_statistic = S2_fact / S2_e

    # Критическое значение для alpha=0.05
    alpha = 0.05
    critical_value = f.ppf(1 - alpha, m, n - m - 1)

    # P-значение
    p_value = 1 - f.cdf(F_statistic, m - 1, n - m)

    return F_statistic, critical_value, p_value


F_statistic, critical_value, p_value = fisher_test(y_test, X_test_mnk, model, len(y_test), X_test_mnk.shape[1] - 1)

print(f"F-критерий: {F_statistic:.3f}")
print(f"Критическое значение: {critical_value:.3f}")
print(f"P-значение: {p_value}")

print()

# Вывод результатов регрессии
result_summary = model.summary()
coefficients_table = pd.DataFrame(result_summary.tables[1].data[1:], columns=result_summary.tables[1].data[0])

# Заменяем имена столбцов
coefficients_table.columns = [' ', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']
print('Анализ статистической значимости коэффициентов уравнения регрессии:')
print(coefficients_table.to_string(index=False))

print()


# Оценка качества модели на тестовых данных
def model_evaluation(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print(f"Среднеквадратичная ошибка на тестовых данных (MSE): {mse:.4f}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Средняя абсолютная ошибка на тестовых данных (MAE): {mae:.4f}")
    R_2 = model.rsquared
    print(f"Коэффициент детерминации: {R_2:.3f}")
    R_2_adj = 1 - (n - 1) / (n - m - 1) * (1 - R_2)
    print(f"Адаптивный коэффициент детерминации: {R_2_adj:.3f}")


model_evaluation(y_test, y_pred)






