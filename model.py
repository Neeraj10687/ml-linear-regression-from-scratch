def loss_func(m, b, points):
    tot_error = 0
    for i in range(len(points)):
        x = points.iloc[i].time
        y = points.iloc[i].score
        tot_error += (y - (m*x + b))**2
    return tot_error / float(len(points))


def grad_descent2(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].time
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b
def grad_descent(m, b, data, L):
    m_grad = 0
    b_grad = 0
    n = len(data)

    for i in range(n):
        x = data.iloc[i].time
        y = data.iloc[i].score
        y_pred = m * x + b

        m_grad += (y_pred - y) * x
        b_grad += (y_pred - y)

    m_grad = (2/n) * m_grad
    b_grad = (2/n) * b_grad

    m = m - L * m_grad
    b = b - L * b_grad

    return m, b
