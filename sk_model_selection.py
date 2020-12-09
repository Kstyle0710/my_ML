'''
model_selection 모듈
   1) 학습용 데이터와 테스트 데이터로 분리
   2) 교차 검증 분할 및 평가
   3) estimator의 하이퍼 파라미터 튜닝을 위한 다양한 함수와 클래스 제공
'''
### train_test_splt() : 학습 및 테스트 데이터 세트 분리########################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.3)
# print(X_train)
# print(y_train)
'''
train_test_split 모듈은 원 데이터를 쪼갠 후 4개의 값을 반환함
변수는 x축 데이터, y축 타겟이며 test_size는 비율임..0.3이면 테스트가 30%, 트레인이 70%
'''
model = LinearRegression()
model.fit(X_train, y_train)

print("학습 데이터 점수 : {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수 : {}".format(model.score(X_test, y_test)))
#
# import matplotlib.pyplot as plt
#
# predicted = model.predict(X_test)   # 위의 학습한 리니어 모델에 X_test 값을 넣을 때 예측 결과값
# expecetd = y_test                  # 실제 사례 test y 값
# plt.figure(figsize = (8, 4))
# plt.scatter(expecetd, predicted)
# plt.plot([0, 350], [0, 350], "--r")
# plt.tight_layout()
# plt.show()

### cross_val_score() : 교차 검증 ###########################################
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

scores = cross_val_score(model, diabetes.data, diabetes.target, cv=5)

print('교차 검증 정확도1 : {}'.format(scores))
print('교차 검증 정확도2 : {} +/- {}'.format(np.mean(scores), np.std(scores)))

### GridSearchCV : 교차 검증과 최적 하이퍼 파라미터 찾기 ##########################
'''
 훈련 단계에서 학습한 파라미터에 영향을 받아서 최상의 파라미터를 찾는 일은 항상 어려운 문제
 다양한 모델의 훈련 과정을 자동화하고, 교차 검사를 사용해 최적 값을 제공하는 도구 필요
'''

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import pandas as pd

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha = alpha)

gs = GridSearchCV(estimator=Ridge(), param_grid = param_grid, cv=10)
result = gs.fit(diabetes.data, diabetes.target)

print("최적 점수 : {}".format(result.best_score_))
print("최적 파라미터 : {}".format(result.best_params_))
print(gs.best_estimator_)
print(pd.DataFrame(result.cv_results_))


### multiprocessing을 이용한 GridSearchCV #########################
import multiprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
param_grid = [{'penalty':['l1', 'l2'],
               'C':[0.5, 1.0, 1.8, 2.0, 2.4]}]
gs = GridSearchCV(estimator=LogisticRegression(), param_grid = param_grid,
                  scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
result = gs.fit(iris.data, iris.target)
print("최적 점수 : {}".format(result.best_score_))
print("최적 파라미터 : {}".format(result.best_params_))
print(gs.best_estimator_)
print(pd.DataFrame(result.cv_results_))



