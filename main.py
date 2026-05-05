import pandas as pd
import argparse
from train import train
from plot import plot_regression, plot_loss, animate_training
from compare import compare_with_sklearn

parser = argparse.ArgumentParser(description="Linear Regression from Scratch")

parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--animate", action="store_true")

args = parser.parse_args()

data = pd.read_csv("data.csv")

# normalize
data["time"] = data["time"] / data["time"].max()
data["score"] = data["score"] / data["score"].max()

print(f"Training with epochs={args.epochs}, lr={args.lr}")

m, b, losses, history = train(data, epochs=args.epochs, L=args.lr)

print(f"Your model: y = {m:.4f}x + {b:.4f}")
compare_with_sklearn(data)

plot_regression(data, m, b)
plot_loss(losses)

if args.animate:
    animate_training(data, history)

