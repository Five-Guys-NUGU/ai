# Normalization - 정규화
# 변수: uid, sex, name, age, gpa, interests, avg_study_time, avg_rest_count
# Dataset: 스스로 제작한 csv_data_example.csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, math, json

def printWithPyplot(df):
    plt.figure(figsize = (10, 8))
    plt.plot(df['total_study_time'], df['avg_study_time'], 'ro', color='lime')
    
    plt.title('Before Normalization')
    #plt.title('After Normalization')
    plt.xlabel('total_study_time')
    plt.ylabel('avg_study_time')
    plt.xticks([0, 2, 4, 6, 8, 10, 12])

    plt.yticks([0, 2, 4, 6, 8, 10, 12])

    plt.show()

# Kmean Clustering을 위한 최적의 K 찾는 함수 - Silhouette Score 이용
def findBestK(df):
    sil = []
    kmax = round(math.sqrt(len(df)))

    # k값을 2 ~ kmax 값까지 테스트 해보기
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters = k).fit(df[['total_study_time', 'avg_study_time']])
        labels = kmeans.labels_
        sil.append(silhouette_score(df[['total_study_time', 'avg_study_time']], labels, metric = 'euclidean'))

    print(sil)

    plt.figure(figsize = (6, 6))
    plt.plot(list(range(2, kmax+1)), sil, color='darkorange')
    
    plt.title('Silhouette Score')
    plt.xlabel('k')
    plt.ylabel('score')
    plt.xticks([2, 3, 4, 5, 6])
    plt.show()

    return (2 + sil.index(max(sil)))

# 입력받은 k를 이용하여 kmean clustering 수행
def kmean(df, k):
    kmeans = KMeans(n_clusters = k).fit(df[['total_study_time', 'avg_study_time']])
    labels = kmeans.labels_

    plt.figure(figsize = (10, 6))
    plt.scatter(df['total_study_time'], df['avg_study_time'], c=labels.astype(float))
    plt.title('K-means Visualization')
    plt.xlabel('total_study_time')
    plt.ylabel('avg_study_time')
    plt.show()

    print(labels)
    return labels

# 점 데이터 분류 - n개의 점 각각에 대한 데이터에서 k개의 데이터로 변환
def distinguish(k, labels):
    lists = []

    for i in range(k):
        lists.append([index for index, value in enumerate(labels) if value == i])

    print(len(lists))
    print(lists)
    return lists
    
# lists에서 매칭하는 함수
# 짝수 개 일때만 고려해서 오류 생길 수 있음 현재
def match(lists):
    couples = []

    for list in lists:
        if len(list) % 2:
            list.pop()
        random.shuffle(list)
        for i in range(0, len(list), 2):
            temp_arr = []
            temp_arr.append(df_uid[list[i]])
            temp_arr.append(df_uid[list[i+1]])
            couples.append(temp_arr)

    return couples


# Read Data from the csv File
df = pd.read_csv('./data/csv_data_example.csv', encoding='euc-kr')
print(df)
df_uid = df['uid'].tolist()
printWithPyplot(df)


# Normalization - 정규화
mmscaler = MinMaxScaler()
df[['total_study_time', 'avg_study_time']] = mmscaler.fit_transform(df[['total_study_time', 'avg_study_time']])
print(df)


k = findBestK(df)
labels = kmean(df, k)
lists = distinguish(k, labels)
print(match(lists))
