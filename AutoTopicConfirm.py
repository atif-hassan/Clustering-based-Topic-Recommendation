import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir

path = "EE Topics Info Only Keywords.txt"
topicsPath = "F:\Programs and Books\Python Projects\Winter Internship\EE\\"




'''--------------------------------Returns the topic closest to the cluster---------------------------------------'''
def getTopics() :
    topics = listdir(topicsPath)
    topicList = []
    
    for topic in topics:
        topic = topic.replace(".txt", "")
        topicList.append(topic)

    return topicList
'''---------------------------------------------------------------------------------------------------------------'''




'''--------------------------------Returns the Distance---------------------------------------'''
def distance(X, index):
    #Calculate the similarity measure(distance). Greater distance means the documents are closer/similar
    cosine = cosine_similarity(X[index], X)

    return cosine
'''-------------------------------------------------------------------------------------------'''




'''------Converts text present at a path into corressponding vector(sparse matrix representation) using tf-idf-----'''
def convertTextToDistanceMatrix(path, topicList):
    topicInfo = []
    #Distance Matrix
    D = [[] for i in range(len(topicList))]

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
    vectorizer = TfidfVectorizer(stop_words=stopSet, use_idf=True, ngram_range=(1, 3))
    X = vectorizer.fit_transform(topicInfo)

    #Create the distance matrix
    for i in range(0, len(topicList)):
        cosine = distance(X, i)
        for dist in cosine[0]:
            D[i].append(dist)

    return D
'''----------------------------------------------------------------------------------------------------------------'''




'''---------------------------------------------------Get Centroids------------------------------------------------'''
def getCentroids(D, topicList, k):
    tmp = []
    centroids = []
    min = 1000
    index = 0

    #Stores the sum of all topics into tmp
    for i in range(0, len(topicList)):
        sum = 0
        for j in range(0, len(topicList)):
            sum+= D[i][j]
        tmp.append([sum, i])

    #Selects the first centroid/topic(least connected with the rest of the topics)
    for i in range(0, len(topicList)):
        if tmp[i][0] < min:
            min = tmp[i][0]
            index = tmp[i][1]
    centroids.append(index)

    for i in range(0, k-1):
        tmp = []
        for j in range(0, len(topicList)):
            sum = 0
            flag = 0
            for m in range(0, len(centroids)):
                if centroids[m] != j:
                    sum+= D[j][centroids[m]]
                else:
                    flag = 1
                    break
            if flag == 0:
                tmp.append([sum, j])
        
        min = 1000
        
        #Selects the centroid/topic which is least connected with the rest of the topics
        for i in range(0, len(tmp)):
            if tmp[i][0] < min:
                min = tmp[i][0]
                index = tmp[i][1]
        centroids.append(index)


    for i in range(0, len(centroids)):
        print(topicList[centroids[i]])

    return centroids
    
'''----------------------------------------------------------------------------------------------------------------'''




'''---------------------------------------------------K Clustering-------------------------------------------------'''
def kClustering(D, topicList, centroids, k):
    #2D list of clusters. Each row corressponds to a single cluster
    clusters = [[] for i in range(k)]

    #List all Centroids in their own clusters
    for i in range(0, k):
        clusters[i].append([topicList[centroids[i]], centroids[i]])

    #For each topic
    for i in range(0, len(topicList)):
        maxDist = -1
        topicIndex = 0
        flag = 0

        #Checks if current topic is a centroid
        for j in range(0, k):
            if topicList[i] == topicList[centroids[j]]:
                flag = 1
                break

        #If not a centroid, then find the best cluster for current topic
        if flag == 0:
            for j in range(0, k):
                index = centroids[j]
                if D[i][index] > maxDist:
                    maxDist = D[i][index]
                    topicIndex = j
        else:
            continue

        #Append the topic
        clusters[topicIndex].append([topicList[i], i])

    print("\n\n\n")

    #Print the clusters
    for clus in clusters:
        print(clus)

    #Select the topic which best represents its cluster
    for l in xrange(0, k):
        storeSum = []
        #Find the sum of each topic with respect to its cluster
        for i in xrange(0, len(clusters[l])):
            index = clusters[l][i][1]
            sums = 0
            for j in xrange(0, len(clusters[l])):
                if index!= j:
                    sums+= D[index][j]
            storeSum.append(sums)

        maxDist = storeSum[0]
        m = 0
        #The best topic is the one whose sum is maximum
        for i in xrange(0, len(storeSum)):
            if maxDist < storeSum[i]:
                maxDist = storeSum[i]
                m = i
        print("\n\n"+clusters[l][m][0])
'''----------------------------------------------------------------------------------------------------------------'''




'''--------------------------------------------------------MAIN----------------------------------------------------'''
def main(k):
    topicList = getTopics()
    #topicList = ["The Exorcist", "Babadook", "The Texas Chainsaw Massacre"]
    D = convertTextToDistanceMatrix(path, topicList)
    centroids = getCentroids(D, topicList, k)
    kClustering(D, topicList, centroids, k)
'''----------------------------------------------------------------------------------------------------------------'''




main(10)
