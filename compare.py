from sklearn.linear_model import LinearRegression

def compare_with_sklearn(data):
    X = data.time.values.reshape(-1, 1)
    y = data.score.values

    model = LinearRegression()
    model.fit(X, y)

    print("Sklearn result:")
    print(f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
