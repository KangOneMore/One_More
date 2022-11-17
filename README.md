Solve the problems Using k-means algorithm
==========================================

## Introduction
 ### What is k-means algorithm
 K-means clustering is one of the simplest and popular unsupervised machine learning algorithms. Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to known, or labelled, outcomes. A cluster refers to a collection of data points aggregated together due to specific similarities, and similarities were measured using the distance to the centroid, i.e., Euclidean distance and RSS.

 Step-1: Select the number K to decide the number of clusters. 
 Step-2: Select random K points or centroids. (It can be other from the input dataset).
 Step-3: Assign each data point to their closest centroid, which will form the predefined K clusters. 
 Step-4: Calculate the RSS value and place a new centroid of each cluster. 
 Step-5: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster. 
 Step-6: If any reassignment occurs, then go to  step-4 else go to FINISH.
 Step-7: The model is ready.

 ### What are the problems
 #### Problem 1: Toy Example 
 The goal is to implement a k-means algorithm and find out how many iterations are needed to find an optimized cluster using a given point and center. Visualize the results at the end of each iteration to see what has changed. To solve the problem, our team was given the following eight points (x,y) :

 and we were also given the three centroids:
 
 #### Problem 2: Real-world Practical Problem 
 A vertiport is a takeoff and landing pad where people can board and exit air vehicle and is essential to advanced Regional Air Mobility (RAM). The construction of vertiports requires careful consideration since a vertiport is subject to land use, noise issues, or public safety so to solve this problem, we used that K-means and Elbow points were used to find the best place using a csv file containing the latitude of candidate 304 vertiport candidate for constructing vertiport.
 

## Installation
 ### ipynb file
 Just run jupyter notebook in terminal and you can visit the notebook in your web browser.
 ```bash
 $ jupyter notebook
 ```
 If you don’t download Jupyter notebook, please install jupyter notebook from [site](https://docs.jupyter.org/en/latest/install.html)

 ### py file
 원모학우님이 적어주세요

## Dependencies
 - Numpy
 - Math
 - matplotlib.pyplot
 - pandas
 - 원모학우님이 추가

 Install missing dependencies using [pip](https://pip.pypa.io/en/stable/) and other [sites](https://datatofish.com/install-package-python-using-pip/)

 if you use jupyter notebook
 ```bash
 $ conda install module_name
 ``` 
 or in jupyter notebook
 ```python
 pip install module_name
 ```
 please refer to [site](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/)


## Code description
### Problem 1: Toy Example
 
 Load the modules needed to execute the code
 ```python
 import numpy as np
 import math
 import matplotlib.pyplot as plt
 import pandas as pd
 %matplotlib inline
 ```

 Make a list of information about samples and centroid
 ```python
 #sample define
 X = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
 X = np.asmatrix(X)
 mu = [[[2,10],[5,8],[1,2]]] # centroid
 ```

 Draw a point plot using sample information
 ```python
 plt.figure(figsize = (6,6))
 plt.plot(X[:,0],X[:,1],'b.',markersize= 15)
 plt.axis('equal')
 plt.show()
 ```

 Draw centroids and samples at once and visually check them
 ```python
 #sample + 3 centroid
 k = 3
 m = X.shape[0]
 a= np.argmax(X, axis = 0)
 initial_mu = np.asmatrix(mu[0]) #length of row

 plt.figure(figsize = (6,6))
 plt.plot(X[:,0],X[:,1],'b.',label = 'given point',  markersize= 10)
 plt.plot(initial_mu[:,0],initial_mu[:,1],'s', color = 'r', markersize= 15,label = 'Centroid')
 plt.axis('equal')
 plt.legend(fontsize = 12)
 plt.show()
 ```

 preparation of k-means algorithm
 ```python
 y = [] # save the nearest cluster for iteration
 pre_mu = mu[0].copy() # copy the initialzed centroid value
 euc_dist_iter = []#store euclidean distance each iteration
 pre_sum = 0 #initialize of sum of previous RSS vlaue
 rss_iter = [] #store RSS values each iteration
 ```
 Implementing the k-means algorithm
 ```python
    for i in range(m): # i means point (m=8)
        y_i = [] 
        d0 = math.sqrt((X[i,0] - pre_mu[0][0])**2 + (X[i,1] - pre_mu[0][1])**2) # distance to the centroid1
        d1 = math.sqrt((X[i,0] - pre_mu[1][0])**2 + (X[i,1] - pre_mu[1][1])**2) # distance to the centroid2
        d2 = math.sqrt((X[i,0] - pre_mu[2][0])**2 + (X[i,1] - pre_mu[2][1])**2) # distance to the centroid3
        y_i = np.argmin([d0,d1,d2]) # y is the cluster number whose centroid is the nearest to the ith point. 
        y_j.append(y_i) # save the nearest cluster of ith point 
        euc_dist.append(np.round([d0,d1,d2],3)) # save the distance between ith point and each centroid.
        new_sum += (np.min([d0,d1,d2]))**2 #calculate RSS values
    
    y.append(y_j) # save the nearest cluster for jth iteration
    euc_dist_iter.append(euc_dist) # save the distance for the jth iteration
    error = pre_sum - new_sum #RSS(j-1th itertaion) - RSS(jth iteration)
    rss_iter.append((j,abs(error)))
    pre_sum = new_sum
  
    # #stop iteration when there's no change of RSS value
    # if error == 0:
    #     break
    

    # Obtain new centroids
    new_mu = []
    
    for z in range(k): # z means centroid (k=3)
        new_mu.append(np.mean(X[np.where(np.array(y_j) == z)[0]],axis = 0).tolist()[0]) 
        # The new centroid is the center of each point to which it is belonged.
        
    pre_mu = new_mu.copy()
    mu.append(new_mu)#add the new centroids to mu list
 ```

 Graph previous RSS - current RSS values obtained through iteration.
 ```python
 rss_iter = np.array(rss_iter) #change list to np.array
 plt.plot(rss_iter[:,0], rss_iter[:,1]) #x axis = number of iterations; y axis = RSS difference(previous - current)
 plt.xlabel('Number of iteration')
 plt.title("RSS value according to iteration")
 plt.ylabel('RSS')
 plt.show()
 plt.savefig('iter_mse.png')
 ```


 Graph of before updating the centroids
 ```python
 #devide cluster
 x0 = X[np.where(np.array(y[0])==0)[0]] #Initial stage cluster1 
 x1 = X[np.where(np.array(y[0])==1)[0]] #Initial stage cluster2
 x2 = X[np.where(np.array(y[0])==2)[0]] #Initial stage cluster3

 #graph of Before updating the centroids
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15) #First cluster
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15) #Second cluster
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15) #third cluster
 plt.plot(mu[0][0][0],mu[0][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') #Initial stage centroid1
 plt.plot(mu[0][1][0],mu[0][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') #Initial stage centroid2
 plt.plot(mu[0][2][0],mu[0][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3') #Initial stage centroid3
 plt.axis("equal")
 plt.title("Before updating the centroids")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```
 
 Graph of result of first iteration with changing cluster
 ```python
 #devide clusters
 x0 = X[np.where(np.array(y[0])==0)[0]] #Initial stage cluster1
 x1 = X[np.where(np.array(y[0])==1)[0]] #Initial stage cluster2
 x2 = X[np.where(np.array(y[0])==2)[0]] #Initial stage cluster3

 #graph of first iteration result
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[1][0][0],mu[1][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') #first changed centroid1
 plt.plot(mu[1][1][0],mu[1][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2')#first changed centroid2
 plt.plot(mu[1][2][0],mu[1][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3')#first changed centroid3
 plt.axis("equal")
 plt.title("1st iteration result")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```

 New cluster reflecting centroids changed by the first iteration
 ```python
 #devide cluster
 x0 = X[np.where(np.array(y[1])==0)[0]] # first changed cluster1
 x1 = X[np.where(np.array(y[1])==1)[0]] # first changed cluster2
 x2 = X[np.where(np.array(y[1])==2)[0]] # first changed cluster3

 #daraw the graph of 1st iteration result with udating clusters
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[1][0][0],mu[1][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') #first changed centroid1
 plt.plot(mu[1][1][0],mu[1][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') #first changed centroid2
 plt.plot(mu[1][2][0],mu[1][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3')#first changed centroid3
 plt.axis("equal")
 plt.title("Before updating the centroid")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```

 Graphs of new centroid and previous clusters obtained through the second iteration
 ```python
 #devide cluster
 x0 = X[np.where(np.array(y[1])==0)[0]] # first changed cluster1
 x1 = X[np.where(np.array(y[1])==1)[0]] # first changed cluster2
 x2 = X[np.where(np.array(y[1])==2)[0]] # first changed cluster3

 #graph of second iteration reuslt
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[2][0][0],mu[2][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') #second changed centroid1
 plt.plot(mu[2][1][0],mu[2][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') #second changed centroid2
 plt.plot(mu[2][2][0],mu[2][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3') #second changed centroid3
 plt.axis("equal")
 plt.title("2nd iteration result")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```

 New centroids obtained through the second iteration and newly clustered clusters based on them
 ```python
 #dividing cluster
 x0 = X[np.where(np.array(y[2])==0)[0]] # second changed cluster1
 x1 = X[np.where(np.array(y[2])==1)[0]] # second changed cluster2
 x2 = X[np.where(np.array(y[2])==2)[0]] # second changed cluster3

 #result of second iteration result
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[2][0][0],mu[2][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') # second changed centroid1
 plt.plot(mu[2][1][0],mu[2][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') # second changed centroid2
 plt.plot(mu[2][2][0],mu[2][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3') # second changed centroid3
 plt.axis("equal")
 plt.title("Before updating the centroid")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```

 Graph of changes in centroid according to the third iteration with previous clusters
 ```python
 #devide cluster
 x0 = X[np.where(np.array(y[2])==0)[0]] # second changed cluster1
 x1 = X[np.where(np.array(y[2])==1)[0]] # second changed cluster2
 x2 = X[np.where(np.array(y[2])==2)[0]] # second changed cluster3

 #graph of third iteration result
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[3][0][0],mu[3][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') # third changed centroid1
 plt.plot(mu[3][1][0],mu[3][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') # third changed centroid2
 plt.plot(mu[3][2][0],mu[3][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3') # third changed centroid3
 plt.axis("equal")
 plt.title("3rd iteration result")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```

 Graph that applies cluster changes according to third changes in centroids
 ```python
 #devide cluster
 x0 = X[np.where(np.array(y[3])==0)[0]] # third changed cluster1
 x1 = X[np.where(np.array(y[3])==1)[0]] # third changed cluster2
 x2 = X[np.where(np.array(y[3])==2)[0]] # third changed cluster3

 #clustering after third iteration
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[3][0][0],mu[3][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') # third changed centroid1
 plt.plot(mu[3][1][0],mu[3][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') # third changed centroid2
 plt.plot(mu[3][2][0],mu[3][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3') # third changed centroid3
 plt.axis("equal")
 plt.title("Before updating the centroid")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```

 Graphs for samples clustered by the third iteration and new centroid changed by the fourth
 ```python
 #devide cluster
 x0 = X[np.where(np.array(y[3])==0)[0]] # third changed cluster1
 x1 = X[np.where(np.array(y[3])==1)[0]] # third changed cluster2
 x2 = X[np.where(np.array(y[3])==2)[0]] # third changed cluster2


 #clustering after 4th iteration
 plt.figure(figsize = (6,6))
 plt.plot()
 plt.plot(x0[:,0],x0[:,1],'b. ',label = 'Cluster 1',markersize= 15)
 plt.plot(x1[:,0],x1[:,1],'g. ',label = 'Cluster 2',markersize= 15)
 plt.plot(x2[:,0],x2[:,1],'r. ',label = 'Cluster 3',markersize= 15)
 plt.plot(mu[4][0][0],mu[4][0][1],'s', color = 'b', markersize= 15,label = 'Centroid 1') # fourth changed centroid1
 plt.plot(mu[4][1][0],mu[4][1][1],'s', color = 'g', markersize= 15,label = 'Centroid 2') # fourth changed centroid2
 plt.plot(mu[4][2][0],mu[4][2][1],'s', color = 'r', markersize= 15,label = 'Centroid 3') # fourth changed centroid3
 plt.axis("equal")
 plt.title("4th iteration result")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.legend(fontsize = 12,loc='center left',bbox_to_anchor=(1, 0.5) )
 plt.show()
 ```




## Authors
 Group 6 of 2022 Fall Semester 

  * j21900432 Jung Sook Yang
  * 21900649 Ka Won Jeong
  * 22001003 Won Mo Kang
  * 22001034 Sang Been Woo

![image](https://user-images.githubusercontent.com/118418436/202356461-9fb80e76-6034-4c62-b66c-f691e3a60950.png)
