from model import grad_descent, loss_func


def train(data, epochs=3000, L=0.02):
    m = 0
    b = 0
    losses = []
    history=[]

    for i in range(epochs):
        m, b = grad_descent(m, b, data, L)
        loss = loss_func(m, b, data)
        losses.append(loss)
        if i % 1000 ==0:
            print(f"{i} m={m:.4f}, b= {b:.4f}")

        if i % 100 == 0:
            print(f"epoch {i}, Loss: {loss}")

        if i % 50 ==0:
            history.append((m,b))

    return m, b, losses,history
