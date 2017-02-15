import nltk
import numpy
import random


#Define the maximum iteration for K-Means Clustering
maxIte = 4

textFilePath = "Research Areas.txt"
newTextFilePath = "Pre-Processed Research Areas.txt"


#Extracts key words to remove noise(like prepositions and conjunctions)
def extractImpWords():
    
    fp = open(textFilePath, "r")
    
    for line in fp:
        #String pre-processing
        text = line.strip()
        '''text = text.replace(',', '')
        text = text.replace('.', '')
        text = text.replace(';', ',')'''
        text = text.lower()

        '''#Split Sentence into each word
        textSplit = nltk.word_tokenize(text)
        #Tag each word with its type in the english language
        words = nltk.pos_tag(textSplit)

        newWords = ""
        count = 0

        #If word is co-ordinating conjuction(CC) or a preposition(IN), then dont account
        for i in range(0, len(textSplit)):
            if 'CC' == words[i][1] or 'IN' == words[i][1]:
                continue
            else:
                newWords+= words[i][0]+" "
                count+=1

        if count > maxSize:
            maxSize = count

        count = 0

        #Append new line to new file
        with open(newTextFilePath, "a") as fp2:
            fp2.write(newWords.strip()+"\n")
            print(newWords.strip()+"\n")'''

        with open(newTextFilePath, "a") as fp2:
            fp2.write(text+"\n")

        

    fp.close()
    fp2.close()



def determineInitialCentroids(k):
    textFilePath = "Research Areas.txt"
    fp = open(textFilePath, "r")

    #All available topics are stored in this list
    topicList = []
    #The frequency of each topic is stored in this list
    freq = []
    #Used for checking
    flag = 0

    #Final chosen centroid(topics) list
    centroids = []

    #Read each line
    for line in fp:
        #Split each line into topics separated by comma
        words = line.strip().split(',')
        #Check if current topic already exists, then increase its count else append it to the list
        for i in range(0, len(words)):
            for j in range(0, len(topicList)):
                if topicList[j] == words[i].strip():
                    freq[j]+=1
                    flag = 1
                    break
            if flag == 0:
                topicList.append(words[i].strip())
                freq.append(1)
            else:
                flag = 0

    #Sort the topics in decreasing order of frequency       <--Requires optimal sorting
    for i in range(0, len(freq)):
        for j in range(0, len(freq)):
            if freq[i] > freq[j]:
                temp = freq[i]
                freq[i] = freq[j]
                freq[j] = temp
                tempStr = topicList[i]
                topicList[i] = topicList[j]
                topicList[j] = tempStr

    #Take the first k topics as centroid
    for i in range(0, k):
        centroids.append(topicList[i])

    #Remove unnecessary words from topics
    for i in range(0, len(centroids)):
        centroids[i] = centroids[i].replace('and ', '')
        centroids[i] = centroids[i].replace('for ', '')
        centroids[i] = centroids[i].lower()

    #Initial Centroids
    print("Initial Centroids : \n")
    for i in range(0, len(centroids)):
        print(centroids[i])

    return centroids




def distance(sent, cent):
    #Sent -> Sentence    Cent -> Centroid
    words = sent.split()
    topics = cent.split()
    freq = 0

    #Finds the max occurence of a topic in a sentence. Returns the frequency
    for i in range(0, len(topics)):
        for j in range(0, len(words)):
            if topics[i].strip() == words[j].strip():
                freq+=1

    return freq



def newCentroid(lines):
    #All available topics are stored in this list
    topicList = []
    #The frequency of each topic is stored in this list
    freq = []
    #Used for checking
    flag = 0
    
    for line in range(0, len(lines)):
        #Split each line into topics separated by comma
        words = lines[line].strip().split(',')
        #Check if current topic already exists, then increase its count else append it to the list
        for i in range(0, len(words)):
            for j in range(0, len(topicList)):
                if topicList[j] == words[i].strip():
                    freq[j]+=1
                    flag = 1
                    break
            if flag == 0:
                topicList.append(words[i].strip())
                freq.append(1)
            else:
                flag = 0

    #Sort the topics in decreasing order of frequency       <--Requires optimal sorting
    for i in range(0, len(freq)):
        for j in range(0, len(freq)):
            if freq[i] > freq[j]:
                temp = freq[i]
                freq[i] = freq[j]
                freq[j] = temp
                tempStr = topicList[i]
                topicList[i] = topicList[j]
                topicList[j] = tempStr

    #Select the first topic which has occurred maximum number of times
    newCentroid = topicList[0]
    newCentroid = newCentroid.replace('and ', '')
    newCentroid = newCentroid.replace('for ', '')
    newCentroid = newCentroid.lower()

    return newCentroid
        
        




def kMeansClustering(k, centroids):
    #Counts the number of iterations
    ite = 1

    while(True):
        #2D list of clusters. Each row corressponds to a single cluster
        clusters = [[] for i in range(k)]
        #Open the pre-processed file
        fp = open(newTextFilePath, "r")

        #Pick each line and find its correct cluster by calculating the centroid which occurs max no. of times in that line
        for line in fp:
            dist = 0
            index = 0
            for i in range(0, k):
                if distance(line.strip(), centroids[i].strip()) > dist:
                    dist = distance(line.strip(), centroids[i].strip())
                    index = i
            clusters[index].append(line.strip())

        #Print the centroid and the corressponding cluster
        for i in range(0, k):
            print("\n"+centroids[i].strip()+" :")
            for j in range(0, len(clusters[i])):
                print(clusters[i][j])


        print("\n\n\n\nIteration "+str(ite)+" Done\n\n\n\n")
        ite+=1

        fp.close()

        if ite > maxIte:
            break
    
        #Gets new centroid from each cluster
        for i in range(0, k):
            clus = clusters[i]
            if len(clus) > 0:
                centroids[i] = newCentroid(clus)


    #Final Centroids
    print("Final Centroids : \n")
    for i in range(0, k):
        print(centroids[i].strip())



#extractImpWords()

centroids = determineInitialCentroids(10)
print("\n\n\n")
kMeansClustering(10, centroids)
