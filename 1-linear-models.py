from sklearn import linear_model
reg = linear_model.LinearRegression()
lr = reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(lr)
print(f"Coefficient: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
