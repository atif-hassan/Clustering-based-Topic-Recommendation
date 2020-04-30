# Clustering based Topic Recommendation
Winter Internship at the Indian Institute of Technology, Kharagpur

# ABSTRACT:
During my internship, I worked on a clustering based recommendation system which was originally used to recommend k topics from a given set of n topics that were put up for display as the varied research areas on offering by IIT Kharagpur for each of their courses. Instead of using the traditional K-Means clustering algorithm which has the pitfalls of randomly choosing initial clusters while being limited to numeric data only and also requiring a certain number of iterations to achieve good results, I used a vector based approach for fixed initial cluster center initialization which allows for repeatable clustering results whilst working with string data and requiring only one iteration to produce the desired final clusters. This system can be used for recommending k subjects, their sub topics or even be extended for any other object, based on the underlying k clusters.

# INTRODUCTION:
Each stream or branch at any institute has a vast number of subjects to offer that a student can pursue as his / her research topic. But not all of these subjects can be uploaded to the institute’s website as its research offering since it makes the website huge and clunky. Usually, teachers of each branch meet up and decide on a minimum number of topics that best represent their stream. The amount of time and manual labour required for such a task becomes huge as all decisions need to be accounted for from each individual professor. Thus, the need of a recommendation system arose which, given a set of topics or subjects, could recommend the required solution.

Clustering methods require inter-cluster distance to be maximum while minimizing intra-cluster distance. In general K-means is a popular partitional clustering algorithm which aims for the same objective. The problem with random center cluster initialization for this particular problem is that if any topics that are similar to each other are chosen as centroids, then the resulting clusters become redundant or insignificant. We require a method wherein the K centroids chosen are topics that are of least similarity.

# METHODS:
**Dataset:**

26 topics from both Computer Science and Engineering along with Electronics and Communication Engineering departments were taken separately for clustering. The similarity measure chosen was the syllabus, taken from IIT Kgp’s Website itself for each topic from both streams as the syllabus consists of topics, sub topics and their description which easily allows the subjects to be distinguished from one another.

**Pre-processing:**

Data for each topic was loaded into python and the nltk package was used to remove stopwords and perform lemmatization.

**Algorithm:**

In order to work with non-numeric data, they must be first converted into meaningful numeric values. **tf–idf**, short for **term frequency–inverse document frequency**, was used to represent the text and **cosine similarity** was used as a measure of similarity between different topics.

A NxN matrix was constructed to keep track of the similarity score between each vector with all other vectors. Now for each vector A_i in the matrix, its similarity score with all other vectors A_j where i ≠ j was summed up and the result was sorted in ascending order. The first K vectors obtained from the sorted list were considered as centroids for the K clusters since they would be the most dis-similar topics. This ensured **high inter cluster distance**. Now for each remaining topic, its corresponding vector was put into the cluster that resulted in the maximum similarity score on comparison with each of the centroids. This ensured **low intra cluster distance.**

Once the clustering was done, K topics were required to be output. Because of the nature of the clusters, it is not necessary that the centroids best represent their respective clusters because such a topic could be present that encompasses all other topics and is thus the best representative of its cluster. To solve this problem, similarity score for each topic against all other topics within a cluster was summed up. This time, the subject with the maximum score means that it is strongly connected with all the other topics and would thus best represent the contents of its cluster. This was iterated over all the K clusters to finally get the recommendation of K topics which best represent the institute’s research variety.
