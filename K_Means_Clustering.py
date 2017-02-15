import numpy as np
import random


#Define the maximum iteration for K-Means Clustering
maxIte = 7
#Define Convergence threshold for centroids
convThr = 0.0000001


#Function to produce random integers between 0 and maxPointVal. These integers serve as points
def randData(maxPointVal, sod):
    data = []
    for i in range(0, sod):
        data.append(random.randint(0, maxPointVal))
    return data


#def eucDist(x1, y1, x2, y2):
    #return sqrt(pow((x2-x1), 2)+pow((y2-y1), 2))


#Rudimentary distance claculation
def dist(x, y):
    return abs(x-y)


#Selects k random centroids between a given range
def randCent(k, maxPointVal):
    centroids = []
    for i in range(0, k):
        centroids.append(random.randint(0, maxPointVal))
    return centroids


#Check if currently calculated centroid has moved. Also chech for number of iterations
def checkConvergence(oldCentroid, newCentroid, ite):
    if(abs(oldCentroid - newCentroid) <= convThr or ite > maxIte):
        return True
    else:
        return False


#Actual K-Means Clustering logic
def kCluster(data, k, maxPointVal):
    #Define an iteration counter
    ite = 0

    #Select random centroids
    centroids = randCent(k, maxPointVal)
    print(centroids)

    #Run an infinite loop
    while(True):
        #Define a variable to calculate minimum distance of a given point from k centroids. Also store the centroid for which minimum value was calculated
        minDist = 100000
        tempK = 100

        #2D list of clusters. Each row corressponds to a single cluster
        clusters = [[] for i in range(k)]

        ##Puts each data point in closest cluster##
        #For each point in data set
        for point in range(0, len(data)):
            #For each centroid
            for i in range(0, k):
                #Calculate optimal cluster for given point
                if(dist(data[point], centroids[i]) < minDist):
                    minDist = dist(data[point], centroids[i])
                    tempK = i
                    
            print("Point "+str(data[point])+" belongs to Cluster : "+str(tempK)+"\n")
            clusters[tempK].append(data[point])
            
            #Reset variable
            minDist = 100000

        #Increase the iteration
        ite+=1

        print(centroids)
        print(clusters)
        print("\n\nIteration "+str(ite)+" complete\n\n")
        
        ##Re-Calculates the centroid of each cluster##
        #For each centroid
        for i in range(0, k):
            tempSum = 0
            #Sum up all points in current cluster
            for j in range(0, len(clusters[i])):
                tempSum+= clusters[i][j]

            #Check if oldCcentroid is close enough to newCentroid
            oldCentroid = centroids[i]
            if(len(clusters[i]) == 0):
                newCentroid = tempSum/1
            else:
                #New centroid is the average of all data points in current cluster
                newCentroid = tempSum/len(clusters[i])

            #If the centroid didn't move much, return
            if(checkConvergence(oldCentroid, newCentroid, ite)):
                return clusters
            else:
                #New centroid is stored
                centroids[i] = newCentroid

            
    return clusters


#Create data
data = randData(100, 20)
print(data)
#Create clusters
clusters = kCluster(data, 3, 100)
