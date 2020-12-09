### 분류 또는 회귀용 데이터 세트 (내장)
'''
   datasets.load_boston()        : 미국 보스턴의 집에 대한 특징과 가격 데이터(회귀용)
   datasets.load_breast_cancer() : 위스콘신 유방암 특징들과 악성/음성 레이블 데이터(분류용)
   datasets.diabetes()           : 당뇨 데이터(회귀용)
   datasets.load_digits()        : 0에서 9까지 숫자 이미지 픽셀 데이터(분류용)
   datasets.load_iris()          : 붓꽃에 대한 특징을 가진 데이터(분류용)
'''
### 일반적인 예제 데이터 세트의 구조
'''
일반적인 예제 데이터 세트의 구조
   일반적으로 딕셔너리 형태로 구성
   data : 특징 데이터 세트 (x 축)
   target : 분류용은 레이블, 회귀용은 숫자 값 데이터 (y 축)
   target_names : 개별 레이블의 이름(분류용)
   feature_names : 특징 이름
   DESCR : 데이터세트에 대한 설명과 각 특징 설명 (디스크립션)
'''

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.keys())
print("--------------")
print(diabetes.data)
print("--------------")
print(diabetes.target)
print("--------------")
print(diabetes.DESCR)
print("--------------")
print(diabetes.feature_names)
print("--------------")
print(diabetes.data_filename)