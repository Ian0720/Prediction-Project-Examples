# 필요한 모듈을 가져옵니다.
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 당뇨병의 데이터를 불러오도록 합니다.
diabetes = datasets.load_diabetes()

# 당뇨병의 데이터셋은 다음과 같습니다.
# Sample Total이 442이며, Dimensionality가 10 이므로
# (442X10) 행렬이 되겠습니다.
# Features(X or data)는 실수이며, -.2에서 .2 사이이고
# Targets(y or target)은 정수이며, 25에서 346사이의 숫자로 이루어져 있습니다.

# 이제 data의 한 열을 골라 X로 써줄 것입니다.
diabetes_X = diabetes.data[:, np.newaxis, 2]
# 해석은 이렇습니다, ':'은 전체 행을 가져오고
# 'np.newaxis'로 차원을 하나 늘려주었습니다.
# 그 다음, '2'로 3번째 열의 데이터를 가져오게 됩니다.
# 2인데 3번째 열을 가져오는 이유는 0부터 세기 때문입니다.
# 그렇다면, 결과적으로 diabetes_X는 (442X1) 행렬이 됩니다.

# diabetes_X를 train set과 test set으로 나누어 주겠습니다.
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# 여기서, train set은 처음부터 -19까지
# test set은 -20부터 끝까지 입니다.

# diabetes_y도 마찬가지로 데이터 셋을 나누어 주겠습니다.
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 그 다음, 선형 회귀 오브젝트를 만들어줍니다.
# 당뇨병 데이터에서 파라미터를 조정해야 할 부분은 딱히 보이지 않습니다.
regr = linear_model.LinearRegression()

# 만든 선형 회귀 오브젝트(모델)에 diabetes_X_test라는 새로운 X를 넣어 y를 예측합니다.
diabetes_y_pred = regr.predict(diabetes_X_test)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
# 모형 결과를 해석해보겠습니다.
# regr.coef_로 회귀 계수를
# mean_squared_error로 MSE 값을
# r2_score로 $R^{2}$ 값을 구할 수 있습니다.
# .2f는 소수점 둘째자리까지 나타내라는 뜻입니다.

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

# plt.scatter는 산포도를 그려줍니다. diabetes_X_test, diabetes_y_test에 해당하는 점을 검정색으로 표시해줍니다.
# plt.plot은 라인 그래프를 그려줍니다. diabetes_X_test, diabetes-Y_pred에 해당하는 선을 파란색으로, 굵기는 3으로 표시합니다.
# plt.xsticks(())와 plt.ysticks(())는 눈금을 나타내는 명령어 입니다.
# 하지만 괄호에 특정한 값을 넣어주지 않았으므로, 표시하지 않게 됩니다.
# 마지막으로, plt.show()는 그래프를 그려 보여줍니다.

# 코드 원문은 다음 주소와 같습니다.
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py