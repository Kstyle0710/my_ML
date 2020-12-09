'''
preprocessing 데이터 전처리 모듈
  데이터의 특징 스케일링을 위한 방법으로 표준화와 정규화를 사용
'''

### StandardScaler : 표준화 클래스 ###############################
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# print(iris_df.describe())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df)
## fit_transform을 하면 결과가 numpy array 형태로 반환되기 때문에 아래에서 다시 dataframe으로 변환시킴
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
# print(iris_df_scaled.describe())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(iris_df_scaled, iris.target, test_size=0.3)
## iris_df_scaled는 전처리된 x축 값인 iris.data임
model = LogisticRegression()
model.fit(X_train, y_train)

print("훈련 데이터 점수 : {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수 : {}".format(model.score(X_test, y_test)))

### MinMaxScaler : 정규화 클래스 ###############################

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
iris_scaled = scaler.fit_transform((iris_df))
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
# print(iris_df_scaled.describe())
X_train, X_test, y_train, y_test = train_test_split(iris_df_scaled, iris.target, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)

print("훈련 데이터 점수1 : {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수1 : {}".format(model.score(X_test, y_test)))