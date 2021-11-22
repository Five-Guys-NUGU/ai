# Normalization - 정규화
# 변수: uid, sex, name, age, gpa, interests, avg_study_time
# Dataset: 스스로 제작한 csv_data_example.csv


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read Data from the csv File
df = pd.read_csv('./data/csv_data_example.csv', encoding='euc-kr')
print(df)

# data = df.loc[:, ['age', 'gpa', 'avg_study_time']]

# Normalization
stdscaler = MinMaxScaler()
df[['avg_study_time', 'avg_rest_count']] = stdscaler.fit_transform(df[['avg_study_time', 'avg_rest_count']])

print(df)

# 화면(figure) 생성
plt.figure(figsize = (10, 6))

# K 값을 늘려가며 반복 테스트
for i in range(1, 7):
    # 클러스터 생성
    estimator = KMeans(n_clusters = i)
    ids = estimator.fit_predict(df[['avg_study_time', 'avg_rest_count']])
    # 2행 3열을 가진 서브플롯 추가 (인덱스 = i)
    plt.subplot(3, 2, i)
    plt.tight_layout()
    # 서브플롯의 라벨링
    plt.title("K value = {}".format(i))
    plt.xlabel('avg_study_time')
    plt.ylabel('avg_rest_count')
    # 클러스터링 그리기
    plt.scatter(df['avg_study_time'], df['avg_rest_count'], c=ids)  
plt.show()
