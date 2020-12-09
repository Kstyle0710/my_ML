### scikit-learn 주요 모듈
'''
scikit-learn 주요 모듈
   sklearn.datasets           : 내장된 예제 데이터 세트
   sklearn.preprocessing      : 다양한 데이터 전처리 기능 제공(변환, 정규화, 스케일링 등)
   sklearn.feature_selection  : 특징을 선택할 수 있는 기능 제공
   sklearn.feature_extraction : 특징 추출에 사용
   sklearn.decomposition      : 차원 축소 관련 알고리즘 지원 (PCA, NMF, Truncated SVD 등)
   sklearn.model_selection    : 교차 검증을 위한 데이터를 학습/테스트용으로 분리, 최적 파라미터를 추출하는 API 제공(GridSearch 등)
   sklearn.metrics            : 분류, 회귀, 클러스터링, Pairwise에 대한 다양한 성능 측정방법 제공
   sklearn.pipeline           : 특징 처리 등의 변환과 ML 알고리즘 학습, 예측 등을 묶어서 실행할 수 있는 유틸 제공
   sklearn.linear_mode        : 선형 회귀, 릿지, 라쏘, 로지스틱 회귀 등 회귀관련 알고리즘과 SGD(Stochastic Gradient Descent) 알고리즘 제공
   sklearn.svm                : 서포트 벡터 머신 알고리즘 제공
   sklearn.neighbors          : 최근접 이웃 알고리즘 제공(K-NN 등)
   sklearn.naive_bayes        : 나이브 베이즈 알고리즘 제공 (가우시안 NB, 다항 분포 NB 등)
   sklearn.tree               : 의사결정 트리 알고리즘 제공
   sklearn.ensemble           : 앙상블 알고리즘 제공 (Random Forest, AdaBoost, GradientBoost 등)
   sklearn.cluster            : 비지도 클러스터링 알고리즘 제공(K-Means, 계층형 클러스터링, DBSCAN 등)
'''
### sklearn API 사용방법
'''
sklearn API 사용방법
   1) Scikit Learn으로부터 적절한 estimator 클래스를 임포트해서 모델의 클래스를 선택
   2) 클래스를 원하는 값으로 인스턴스화해서 모델의 하이퍼 파라미타 선택
   3) 데이터를 특징 배열과 대상 벡터로 배치
   4) 모델 인스턴스의 fit() 메서드를 호출해 모델을 데이터에 적합
   5) 모델을 새 데이터에 대해 적용
       - 지도학습 : 대체로 predict() 메서드를 사용해 알려지지 않은 데이터에 대한 레이블 예측
       - 비지도학습 : 대체로 transform()이나 predict() 메서드를 사용해 데이터의 속성을 변환하거나 추론
'''



### API 기본 사용방법 연습
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

## 데이터 생성
x = np.random.rand(50) * 10
y = 2 * x + np.random.rand(50)
# plt.scatter(x, y)
# plt.show()

## 1) Scikit Learn으로부터 적절한 estimator 클래스를 임포트해서 모델의 클래스를 선택
from sklearn.linear_model import LinearRegression

## 2) 클래스를 원하는 값으로 인스턴스화해서 모델의 하이퍼 파라미타 선택
model = LinearRegression(fit_intercept=True)

## 3) 데이터를 특징 배열과 대상 벡터로 배치
# print(x)  # 1차원 array 형태로 출력됨
X = x[:, np.newaxis]
# print(X)  # 축을 추가해서 2차원 array 형태로 출력됨

## 4) 모델 인스턴스의 fit() 메서드를 호출해 모델을 데이터에 적합
model.fit(X, y)
# print(model.coef_)

## 5) 모델을 새 데이터에 대해 적용
xfit = np.linspace(-1, 11)
# print(xfit)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
print(yfit)

plt.scatter(x, y)    # 원 데이터 분포
plt.plot(xfit, yfit, '--r')   # 모델 예측
plt.show()







