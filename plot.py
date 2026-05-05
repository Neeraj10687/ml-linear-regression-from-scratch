import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def animate_training(data, history):
    fig, ax = plt.subplots()

    # scatter data
    ax.scatter(data.time, data.score)

    # line that will update
    x_vals = np.linspace(data.time.min(), data.time.max(), 100)
    line, = ax.plot([], [], color='red')

    def update(frame):
        m, b = history[frame]
        y_vals = m * x_vals + b
        line.set_data(x_vals, y_vals)
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=100,
        blit=True
    )

    plt.title("Training Animation")
    ani.save("plots/training.gif", writer="pillow")
    plt.show()
def plot_regression(data, m, b):
    plt.scatter(data.time, data.score)
    x_vals = np.linspace(data.time.min(),data.time.max(),100)
    plt.plot(x_vals, m * x_vals + b)
    plt.title("Regression Line")
    plt.savefig("plots/regression.png")
    plt.show()


def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("plots/loss.png")
    plt.show()
