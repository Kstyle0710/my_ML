### 성능 평가 지표
'''
정확도(Accuracy)
  정확도는 전체 예측 데이터 건수 중 예측 결과가 동일한 데이터 건수로 계산
  scikit learn 에서는 accuracy_score 함수를 제공
'''

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

print("훈련 데이터 점수 : {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수 : {}".format(model.score(X_test, y_test)))

from sklearn.metrics import accuracy_score
predict = model.predict(X_test)
print("정확도 : {}".format(accuracy_score(y_test, predict)))
## 결과 자체는 위의 평가 데이터 점수와 동일하다.

'''
오차 행렬(Confusion Maxtix)
   True Negative : 예측값을 Negative 값 0으로 예측했고, 실제 값도 Negative 값 0
   False Positive : 예측값을 Positive 값 1로 예측했는데, 실제값은 Negative 값 0
   False Negative : 예측값을 Negative 값 0으로 예측했는데, 실제값은 Positive 값 1
   True Positive : 예측값을 Positive 값 1로 예측했고, 실제 값도 Positive 값 1
'''
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=predict)
print(confmat)

import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
# plt.xlabel("Predicted label")
# plt.ylabel("True label")
# plt.tight_layout()
# plt.show()

'''
정밀도(Precision)와 재현율(recall)
   정밀도 = TP / (FP+TP)
   재현율 = TP / (FN+TP)
   정확도 = (TN + TP) / (TN+FP+FN+TP)
   오류율 = (FN + FP) / (TN+FP+FN+TP)
'''
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
print("정밀도 : {}".format(precision))
print("재현율 : {}".format(recall))

'''
F1 Score(F-measure)
   정밀도와 재현율을 결합한 지표
   정밀도와 재현율이 어느 한쪽으로 치우치지 않을 때 높은 값을 가짐
'''
from sklearn.metrics import f1_score
f1 = f1_score(y_test, predict)
print("F1 Score : {}".format(f1))

'''
ROC 곡선과 AUC
   ROC 곡선은 FPR(False Positive Rate)이 변할 때, TPR(True Positive Rate)이 어떻게 변하는지 나타내는 곡선
   AUC (Area Under Curve) 값은 ROC 곡선 밑에 면적을 구한 값 (1에 가까울수록 좋은 값)
'''
from sklearn.metrics import roc_curve
import numpy as np
pred_proba_class1 = model.predict_proba(X_test)[:, 1]
fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)

plt.plot(fprs, tprs, label='ROC')
plt.plot([0, 1], [0, 1], '--k', label='Random')
start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('FPR(1-Sensitivity)')
plt.ylabel('TPR(Recall)')
plt.legend()
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc =  roc_auc_score(y_test, predict)
print("ROC AUC SCORE: {}".format(roc_auc))



