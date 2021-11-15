# Linear Regression
# 독립변수: study_hours, 종속변수: gpa
# Dataset: https://www.openintro.org/data/index.php?data=gpa_study_hours

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read Data from the csv File
df = pd.read_csv('./data/gpa_study_hours.csv')
print(df)

x = df['study_hours']
y = df['gpa']
plt.plot(x, y, 'o')
line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1, 1), y)
plt.plot(x,line_fitter.predict(x.values.reshape(-1,1)))
plt.show()

print()
print(f"Equation between study_hours and gpa is 'y = {line_fitter.coef_[0]}x + {line_fitter.intercept_}'.")
print('x = study_hours, y = gpa')

# 30시간 공부한 학생이 받을 점수 예측
# print(line_fitter.predict([[30]]))