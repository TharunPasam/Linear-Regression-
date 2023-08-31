import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('Interview.csv')
df

reg = linear_model.LinearRegression()
reg.fit(df[['Experience','Assessment marks (Out of 10)','Interview marks(Out of 10)']], df.Salary)

reg.coef_

reg.intercept_

reg.predict([[6, 6, 6.0]])

