import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

def loss_func(m,b,points):
    tot_error=0
    for i  in range(len(points)):
        x=points.iloc[i].time
        y=points.iloc[i].score
        tot_error +=(y - (m*x+b))**2
    return tot_error/float(len(points))

def grad_descent(m_now,b_now,points,L):
    m_gradient=0
    b_gradient=0

    n=len(points)

    for i in range(n):
        x= points.iloc[i].time
        y= points.iloc[i].score

        m_gradient += -(2/n)*x*(y-(m_now * x +b_now))
        b_gradient += -(2/n)*(y-(m_now * x +b_now))

    m=  m_now - m_gradient*L
    b=  b_now - b_gradient*L
    return m,b

m = 0
b = 0

L=0.0001
epochs=2000
losses=[]

for i in range(epochs):
    m,b=grad_descent(m,b,data,L)
    loss=loss_func(m,b,data)
    losses.append(loss)

    if i % 100 ==0:
        print (f"epoch {i}, Loss: {loss}")

print(m,b)

plt.scatter(data.time,data.score,color="black")
plt.plot(list(range(20,80)),[m *x +b for x in range(20,80)],color="red")
plt.show()

plt.plot(losses)
plt.title("loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
