from topia.termextract import extract
import nltk
from nltk.corpus import stopwords
from os import listdir


path = "F:\Programs and Books\Python Projects\Winter Internship\CSE\\"
extractor = extract.TermExtractor()

pathStore = "F:\Programs and Books\Python Projects\Winter Internship\CSE Topics Info Only Keywords.txt"
fpStore = open(pathStore, "a")

docs = listdir(path)
for doc in docs:
    fp = open(path+doc, "r")
    text = fp.readlines()
    s = ""
    m = ""
    for info in text:
        info = info.lower()
        info = info.replace("\n", "")
        s+= info+" "
    #To store keywords
    k = sorted(extractor(s))
    print(doc+":\n")
    for i in k:
        fpStore.write(i[0]+" ")
        print(i[0]+" ")
    fpStore.write("\n")
    print("\n\n\n")

    #To store original content
    '''print(doc+":\n")
    fpStore.write(s)
    print(s)
    fpStore.write("\n")
    print("\n\n\n")'''
    
fp.close()
fpStore.close()
