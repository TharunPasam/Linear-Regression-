import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv('houseprices.csv')
df

%matplotlib inline
plt.xlabel('AREA')
plt.ylabel('PRICE')
plt.scatter(df.AREA,df.PRICE,color='red',marker='+')

reg = linear_model.LinearRegression ()
reg.fit(df[['AREA']],df.PRICE)

reg.predict([[2900]])

reg.coef_

reg.intercept_

%matplotlib inline
plt.xlabel('AREA', fontsize=20)
plt.ylabel('PRICE', fontsize=20)
plt.scatter(df.AREA, df.PRICE, color='red', marker='+')
plt.plot(df.AREA,reg.predict(df[['AREA']]), color ='blue')

d = pd.read_csv('areas.csv')
d.head(6)

p = reg.predict(d)

d['prices'] = p
d

d.to_excel('prediction.xlsx', index=False)