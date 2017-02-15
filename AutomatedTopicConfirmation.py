import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import random
from sklearn.decomposition import TruncatedSVD

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np




textFilePath = "Research Areas.txt"
#26 unique topics For CSE



'''-----------------------Gives a list of all available unique topics---------------------------------------------'''
def getAllTopics():
    textFilePath = "Research Areas.txt"
    fp = open(textFilePath, "r")

    #All available topics are stored in this list
    topicList = []
    #Used for checking
    flag = 0

    #Read each line
    for line in fp:
        #Split each line into topics separated by comma
        words = line.strip().split(',')
        #Check if current topic already exists, then increase its count else append it to the list
        for i in range(0, len(words)):
            words[i] = words[i].strip().lower()
            for j in range(0, len(topicList)):
                if topicList[j] == words[i]:
                    flag = 1
                    break
            if flag == 0:
                topicList.append(words[i])
            else:
                flag = 0

    return topicList
'''---------------------------------------------------------------------------------------------------------------'''




'''--------------------------------Returns the topic closest to the cluster---------------------------------------'''
def distance(X, index):
    #Calculate the similarity measure(distance). Greater distance means the documents are closer/similar
    cosine = cosine_similarity(X[index], X)

    return cosine
'''---------------------------------------------------------------------------------------------------------------'''




'''---------------------------Select K Random Centroids from the given topic list---------------------------------'''
def selectKcentroids(topics, k):
    '''textFilePath = "Research Areas.txt"
    fp = open(textFilePath, "r")

    #All available topics are stored in this list
    topicList = []
    #Used for checking
    flag = 0

    centroids = []

    
    #Read each line
    for line in fp:
        #Split each line into topics separated by comma
        words = line.strip().split(',')
        #Check if current topic already exists, then increase its count else append it to the list
        for i in range(0, len(words)):
            words[i] = words[i].strip().lower()
            for j in range(0, len(topicList)):
                if topicList[j][0] == words[i]:
                    flag = 1
                    topicList[j][1]+= 1
            if flag == 0:
                topicList.append([words[i], 0])
            else:
                flag = 0
            
    for i in range(0, len(topicList)):
        for j in range(0, len(topicList)):
            if topicList[i][1] > topicList[j][1]:
                topicList[i], topicList[j] = topicList[j], topicList[i]

    for i in range(0, k):
        for j in range(0, len(topics)):
            if topicList[i][0] == topics[j]:
                centroids.append(j)
                break

    return centroids'''
    
    return random.sample(range(0, len(topicList)-1), k)
'''---------------------------------------------------------------------------------------------------------------'''





'''-----Converts text present at a path into corressponding vector(sparse matrix representation) using tf-idf-----'''
def convertTextToMatrix(path, topicList):
    topicInfo = []

    #Read all paragraphs/text separated by new-line character
    lines = (line.rstrip('\n') for line in open(path))
    #Store each topic info/paragraph/text into array
    for info in lines:
        tmp = info.decode('windows-1252')
        tmp = tmp.lower()
        topicInfo.append(tmp)

    #Defines a set of stop-words which exist in the english language(A set is a collection of non-repeating objects)
    stopSet = set(stopwords.words('english'))

    #Get the tf-idf vectorizer object and convert all documents(Each index of array corressponds to a single document) into respective high-dimensional vector
    vectorizer = TfidfVectorizer(stop_words=stopSet, use_idf=True, ngram_range=(1, 1))
    X = vectorizer.fit_transform(topicInfo)

    return X
'''---------------------------------------------------------------------------------------------------------------'''




'''-------------------------------------------Heirarchial Clustering----------------------------------------------'''
def hClustering(X, topicList):
    #Distance Matrix
    D = [[] for i in range(len(topicList))]
    #Cluster which will contain the entire heirarchy of clusters/tree
    cluster = []

    #Create the distance matrix
    for i in range(0, len(topicList)):
        cosine = distance(X, i)
        for dist in cosine[0]:
            D[i].append(dist)

    #Change to numpy array(allows deletion of rows and columns)
    D = np.asarray(D)
    topicList = np.asarray(topicList)

    #Append first topic
    #cluster.append(topicList[0])



    '''#For each topic present. find the most similar topic, then sum up the distance of both topics to find out which is a subset of which. Remove the subset. Less distance means subset
    for i in range(0, len(topicList)):
        maxDist = -1
        index = 0
        distSum1 = 0
        distSum2 = 0
        indexToDelete = 0
        indexNotDelete = 0
        for j in range(0, len(topicList)):
            if D[i][j] > maxDist and i!=j:
                maxDist = D[i][j]
                index = j

        for j in range(0, len(topicList)):
            distSum1+= D[i][j]
            distSum2+= D[index][j]


        if distSum1 >= distSum2:
            indexToDelete = index
            indexNotDelete = i
        else:
            indexToDelete = i
            indexNotDelete = index

        D = np.delete(D, indexToDelete, 0)
        D = np.delete(D, indexToDelete, 1)

        print("\n\nTopic Deleted : "+topicList[indexToDelete]+"\t\tTopic Not Deleted : "+topicList[indexNotDelete]+"\n\n")

        topicList = np.delete(topicList, indexToDelete)

        if i == len(topicList)-1:
            break'''


    for i in range(0, len(topicList)):
        print("\n\n"+topicList[i]+" :\n")
        distSum1 = 0
        distSum2 = 0
        indexToDelete = 0
        for j in range(0, len(topicList)):
            if D[i][j] >= 0.05 and i!=j:
                for l in range(0, len(topicList)):
                    distSum1+= D[i][l]
                    distSum2+= D[j][l]
                if distSum1 > distSum2:
                    indexToDelete = j
                else:
                    indexToDelete = i
            
                print(topicList[indexToDelete]+"\n")
                D = np.delete(D, indexToDelete, 0)
                D = np.delete(D, indexToDelete, 1)
                topicList = np.delete(topicList, indexToDelete)

                if indexToDelete == i:
                    break
                
            if j >= len(topicList)-1:
                break

        if i == len(topicList)-1:
            break
    


    '''while len(D) > 1:
        maxDist = 2
        index = 0

        #Find out the most similar topic
        for i in range(0, len(D[0])):
            if D[0][i] < maxDist:# and D[0][i]<=0.999999999999999:
                maxDist = D[0][i]
                index = i

        #Row wise updation
        for i in range(0, len(D[0])):
            if i == index:
                continue
            if D[0][i] > D[index][i]:
                D[0][i] = D[index][i]

        #Column wise updation
        for i in range(0, len(D[0])):
            if i == index:
                continue
            if D[i][0] > D[i][index]:
                D[i][0] = D[i][index]

        #Delete copy of the Row which was merged
        D = np.delete(D, index, 0)
        #Delete copy of the Column which was merged
        D = np.delete(D, index, 1)

        #Append the newly merged cluster
        cluster.append(topicList[index])
        #Delete the copy of the cluster which was merged
        topicList = np.delete(topicList, index)


        print("\n\n\n\n")
        print(cluster)'''

    '''count = 0

    while len(D) > 1: 
        for n in range(0, len(topicList)):
            maxDist = 2
            index = 0

            #Find out the most dis-similar topic
            for i in range(0, len(topicList)):
                if D[n][i] < maxDist:# and D[n][i]<=0.999999999999999:
                    maxDist = D[n][i]
                    index = i

            #Row wise updation
            for i in range(0, len(topicList)):
                if i == index:
                    continue
                if D[n][i] > D[index][i]:
                    D[n][i] = D[index][i]

            #Column wise updation
            for i in range(0, len(topicList)):
                if i == index:
                    continue
                if D[i][n] > D[i][index]:
                    D[i][n] = D[i][index]

            #Delete copy of the Row which was merged
            D = np.delete(D, index, 0)
            #Delete copy of the Column which was merged
            D = np.delete(D, index, 1)

            #print(topicList[n]+"\t\t"+topicList[index])

            #Append the newly merged cluster
            if count == 0:
                cluster.append(topicList[index])
            else:
                cluster[n].append([topicList[index], topicList[n]])
            #Delete the copy of the cluster which was merged
            topicList = np.delete(topicList, index)
            #topicList = np.delete(topicList, n)

            for clus in cluster:
                print(clus)


            if n == len(topicList)-1:
                break
        print("\n\n\n\nDONE !!!!")
        count+=1
    print(cluster[0])'''

    print(topicList)





    
'''---------------------------------------------------------------------------------------------------------------'''




'''---------------------------------------------K Means Clustering------------------------------------------------'''
def kMeansClustering(X, topicList):
    #No. of Centroids
    k = 10

    #Distance Matrix
    D = [[] for i in range(len(topicList))]

    #distForEachClust = []

    #Select k random centroids
    centroids = selectKcentroids(topicList, k)

    #Create the distance matrix
    for i in range(0, len(topicList)):
        cosine = distance(X, i)
        for dist in cosine[0]:
            D[i].append(dist)
    print D

    #Change to numpy array(allows deletion of rows and columns)
    #D = np.asarray(D)
    #topicList = np.asarray(topicList)

    ite = 1

    
    while ite<=4:
        #2D list of clusters. Each row corressponds to a single cluster
        clusters = [[] for i in range(k)]

        #List all Centroids in their own clusters
        for i in range(0, k):
            clusters[i].append([topicList[centroids[i]], centroids[i]])

        print(centroids)

        #For each topic
        for i in range(0, len(topicList)):
            maxDist = -1
            topicIndex = 0
            flag = 0


            for j in range(0, k):
                if topicList[i] == topicList[centroids[j]]:
                    flag = 1
                    break

            if flag == 0:
                for j in range(0, k):
                    index = centroids[j]
                    if D[i][index] > maxDist:
                        maxDist = D[i][index]
                        topicIndex = j
            else:
                continue

            clusters[topicIndex].append([topicList[i], i])

        print("\n\n\n")

        for clus in clusters:
            print(clus)



        for l in range(0, k):
            storeSum = []
            for i in range(0, len(clusters[l])):
                index = clusters[l][i][1]
                sum = 0
                for j in range(0, len(topicList)):
                    sum+= D[index][j]
                storeSum.append(sum)

            minDist = storeSum[0]
            m = 0
            for i in range(0, len(storeSum)):
                if minDist > storeSum[i]:
                    minDist = storeSum[i]
                    m = i

            print("\n\n"+clusters[l][m][0])

            centroids[l] = clusters[l][m][1]

        print(centroids)
        ite+=1






    for l in range(0, k):
        storeSum = []
        for i in range(0, len(clusters[l])):
            index = clusters[l][i][1]
            sum = 0
            for j in range(0, len(topicList)):
                sum+= D[index][j]
            storeSum.append(sum)

        maxDist = storeSum[0]
        m = 0
        for i in range(0, len(storeSum)):
            if maxDist < storeSum[i]:
                maxDist = storeSum[i]
                m = i

        print("\n\n"+clusters[l][m][0])






        

    '''#For each topic present
    for i in range(0, len(topicList)):
        flag = 0
        dist = -1

        
        
        #Calculate current topic's similarity measure with all other topics
        cosine = distance(X, i)

        #Check if current topic is not the centroid
        for j in range(0, len(centroids)):
            if topicList[i] == topicList[centroids[j]]:
                flag = 1
                break

        #If current topic is not the centroid, then find which centroid is most similar to current topic
        if flag == 0:
            for j in range(0, len(centroids)):
                if cosine[0][centroids[j]] > dist:
                    dist = cosine[0][centroids[j]]
                    m = j
        else:
            continue

        #Append the topic to its respective cluster
        clusters[m].append([topicList[i], dist, i])

    print("\n\n\n")

    for clus in clusters:
        print(clus)'''



    '''#Calculates total distance of a centroid from all points in its cluster
    for i in clusters:
        totalDist = 0
        for j in i:
            totalDist += j[1]
        distForEachClust.append(totalDist)
        break
    print(distForEachClust)



    newDistForEachClust = []



    for i in range(0, len(centroids)):
        for j in clusters[i]:
            index = j[2]
            cosine = distance(X, index)
            totalDist = 0
            totalDist += cosine[0][centroids[i]]
            for m in clusters[i]:
                if m[2]!= index:
                    totalDist += cosine[0][m[2]]
            newDistForEachClust.append(totalDist)
        break
    print(newDistForEachClust)'''
    


    
        

    '''for i in range(0, k):
        print("\n\n"+topicList[centroids[i]])
        print(clusters[i])'''
        
'''---------------------------------------------------------------------------------------------------------------'''







topicList = getAllTopics()
'''for topic in topicList:
    print(topic)'''
#distanceMatrix(topicList)
X = convertTextToMatrix("F:\Programs and Books\Python Projects\Winter Internship\Key Words.txt", topicList)

'''topicInfo = []
path = "F:\Programs and Books\Python Projects\Winter Internship\Crawled Info On All Research Topics CSE.txt"

#Read all paragraphs/text separated by new-line character
lines = (line.rstrip('\n') for line in open(path))
#Store each topic info/paragraph/text into array
for info in lines:
    tmp = info.decode('windows-1252')
    tmp = tmp.lower()
    topicInfo.append(tmp)

for i in range(0, len(topicList)):
    print(topicList[i]+" :")
    print(topicInfo[i]+"\n\n")'''

kMeansClustering(X, topicList)
#hClustering(X, topicList)
