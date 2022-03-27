
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale 
from sklearn.metrics import silhouette_score


def elbow_method(data, kmax): #Elbow function is to find the optimial k/
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(data)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(data)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(data)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (data[i, 0] - curr_center[0]) ** 2 + (data[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse





df=pd.read_csv("adult.data",sep=",",header=None,
                                        names=['age','workclass','fnlwgt','education'
                                        ,'education-num','marital-status','occupation'
                                        ,'relationship','race','sex','capital-gain','capital-loss'
                                        ,'hours-per-week','native-country','Dominican-Republic','cluster'])

# plt.scatter(df['age'],df['fnlwgt'])
# plt.show()

df=df[['age','fnlwgt']]
data=scale(df[['age','fnlwgt']])

K=10
y=elbow_method(data,K)
x=np.arange(1,K+1,1)
plt.plot(x,y)
plt.title("Find optimal k")
plt.xlabel("K")
plt.ylabel("Err")
plt.show()


cluster=KMeans(n_clusters=3).fit_predict(data)
df['cluster']=cluster # Optimal k from above 

