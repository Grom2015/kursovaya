import numpy as np

def dcca_coefficient(x, y, lag):
    """
    ρ_DCCA (Podobnik & Stanley, 2008)
    x, y – равной длины 1-D массивы; lag – окно.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must be same length")
    if lag >= n:
        raise ValueError("lag must be < length of series")

    # интегрированные (кумулятивные) ряды
    X = np.cumsum(x - x.mean())
    Y = np.cumsum(y - y.mean())

    m   = n // lag                       # количество неперекрывающихся окон
    idx = np.arange(m * lag).reshape(m, lag)

    Fxy = Fxx = Fyy = 0.0
    for seg in idx:
        t = np.arange(lag)
        trend_x = np.polyval(np.polyfit(t, X[seg], 1), t)
        trend_y = np.polyval(np.polyfit(t, Y[seg], 1), t)

        eps_x = X[seg] - trend_x
        eps_y = Y[seg] - trend_y

        Fxy += np.mean(eps_x * eps_y)
        Fxx += np.mean(eps_x ** 2)
        Fyy += np.mean(eps_y ** 2)

    return Fxy / np.sqrt(Fxx * Fyy)
