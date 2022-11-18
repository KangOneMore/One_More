Solve the problems Using k-means algorithm
==========================================

---

## Introduction
 ### 1. What is k-means algorithm
 K-means clustering is one of the simplest and popular unsupervised machine learning algorithms. Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to known, or labelled, outcomes. A cluster refers to a collection of data points aggregated together due to specific similarities, and similarities were measured using the distance to the centroid, i.e., Euclidean distance and RSS.

 - Step-1: Select the number K to decide the number of clusters. 
 - Step-2: Select random K points or centroids. (It can be other from the input dataset).
 - Step-3: Assign each data point to their closest centroid, which will form the predefined K clusters. 
 - Step-4: Calculate the RSS value and place a new centroid of each cluster. 
 - Step-5: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster. 
 - Step-6: If any reassignment occurs, then go to  step-4 else go to FINISH.
 - Step-7: The model is ready.
 
 <img src="https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png" width="400" height="400">

### 2. What are the problems
 #### 1) Problem 1: Toy Example 
 The goal is to implement a k-means algorithm and find out how many iterations are needed to find an optimized cluster using a given point and center. Visualize the results at the end of each iteration to see what has changed. To solve the problem, our team was given the following eight points (x,y) :
 
 point  |  x  | y
------- | --- | ---
point 1 | 2 | 10
point 2 | 2 | 5
point 1 | 8 | 4
point 2 | 5 | 8
point 1 | 7 | 5
point 2 | 6 | 4
point 1 | 1 | 2
point 2 | 4 | 9


 and we were also given the three centroids:
 
 Centriod  |     x     |     y
---------- | --------- | ---------
Centroid 1 |     2     |     10
Centroid 2 |     5     |     8
Centroid 3 |     1     |     2

 
  #### 2) Problem 2: Real-world Practical Problem 
 A vertiport is a takeoff and landing pad where people can board and exit air vehicle and is essential to advanced Regional Air Mobility (RAM). The construction of vertiports requires careful consideration since a vertiport is subject to land use, noise issues, or public safety so to solve this problem, we used that K-means and Elbow points were used to find the best place using a csv file containing the latitude, longitude of candidate 304 vertiport candidate for constructing vertiport.
 <img src="https://www.arbin.com/wp-content/uploads/2021/01/Urban-Air-Mobility-project-eVTOL-flying-above-Paris-iStock-768x549.jpg" width="300" height="300">


--- 

## Usage and Installation
### 1. ipynb file for first problem
 open problem1.ipynb file
  - Just run jupyter notebook in terminal and you can visit the notebook in your web browser.
  ```bash
  $ jupyter notebook
  ```
 
 - If you don’t download Jupyter notebook, please install jupyter notebook from [site](https://docs.jupyter.org/en/latest/install.html)
 - open ML_project_prob1.ipynb file

### 2. ipynb file for second problem 
 open problem2.ipynb file  
 
 Files Required to Run problem2.ipynb file
   - Vertiport_candidates.csv
   - South_Korea_territory.csv
   - korea.json
   - In order to execute the code, the South_Korea_territory.csv file had to be replaced with the json file. I changed the South_Korea_territory.csv file to json file using the following code.

   ```python
   # change csv to json
   f = open('output.txt', 'w')
for i in range(len(korea)):
    abc = korea.loc[i]
    ff.write('[' + str(abc['Longitude (deg)']) + ', ' + str(abc['Latitude (deg)']) + '], ')
   ```
   

--- 

## Dependencies
 - Numpy
 - Math
 - matplotlib.pyplot
 - pandas
 - folium
 - json

 Install missing dependencies using [pip](https://pip.pypa.io/en/stable/) and other [sites](https://datatofish.com/install-package-python-using-pip/)

 - When you download modules in terminal
   ```bash
   $ conda install module_name
   ``` 
   if you have any problem, please refer to [site](https://harlequink.tistory.com/48)
 
 
 - or When you download modules in jupyter notebook
   ```python
   pip install module_name
   ```
 if you have any problem, please refer to [site](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/)
 

---


## Authors
 Group 6 of Machine learning class in 2022 Fall Semester 

  * 21900432 Jung Sook Yang
  * 21900649 Ka Won Jeong
  * 22001003 Won Mo Kang
  * 22001034 Sang Been Woo
## Reference
Change csv to json
[site](https://ko.wikipedia.org/wiki/GeoJSON#cite_note-11)

[site](https://stackoverflow.com/questions/48586647/python-script-to-convert-csv-to-geojson)

[site](https://teddylee777.github.io/visualization/folium)
