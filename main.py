import nltk
import re
import string
import sklearn
import pandas as pd
import itertools as it
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option("display.max_rows", None, "display.max_columns", None)
documents = {'originaldoc_1': [1], 'originaldoc_2': [2], 'originaldoc_3': [3], 'originaldoc_4': [4], 'originaldoc_5': [5]
         }
plagiairised_document = "altered1_doc1"
def paragraphs (fileobj, seperator = '/n'):
    #iterate a fileobj by paragraph
    lines = []
    for line in fileobj:
        if line == seperator and lines:
            yield ''.join(lines)
            lines = []
        else:
            lines.append(line)
    yield ''.join(lines)
def reading_in_files(filenames):
    paragraph_lists = [[]]
    with open(f'data/{filenames}.txt') as f:
        paras = paragraphs(f)
        for para, group in zip(paras, it.cycle(paragraph_lists)):
            group.append(para)
    return paragraph_lists
    # return '\n'.join(strings)
def remove_numbers(text):
    text_nonum = re.sub(r'\d+', '', text)
    return text_nonum

def tagmaker(splitdocument,paragraphnum):
    tempnum = paragraphnum
    paralist = []
    l = 0
    document_taglist = []
    while l<= tempnum:
        plaglabel = "paragraph" + str(l+1)
        paralist.append(plaglabel)
        l = l+1
    for item in splitdocument:
        remove_numbers(item)
        tokens = nltk.word_tokenize(item)
        applytags = nltk.pos_tag(tokens)
        words,taglist = zip(*applytags)
        tagstring = " "
        for j in taglist:
            if j == 'NN':
                tagstring = tagstring + j + " "
            elif j == 'NNS':
                tagstring = tagstring + j + " "
        document_taglist.append(tagstring)
    return paralist,document_taglist

for files,authornumber in documents.items():
    docs = reading_in_files(files)
    plagdoc = reading_in_files(plagiairised_document)
    print("Document in question: " + plagiairised_document)
    print("Document being compared against: " + files)
    for doc in plagdoc[0]:
        plagparanumber = doc.count("\n")
        plag_paragraphsplit = doc.split("\n")
        plagparalist,tag_plag_list = tagmaker(plag_paragraphsplit,plagparanumber)
        plag_vectorizer = TfidfVectorizer()
        plagtfidfMat = plag_vectorizer.fit_transform(tag_plag_list)
        plag_featurenames = sorted(plag_vectorizer.get_feature_names())
        plagSk = pd.DataFrame(plagtfidfMat.todense(), index=(plagparalist), columns=plag_featurenames)

    for i in docs[0]:
        #print(i)
        paranumber = i.count("\n")
        paragraphsplit = i.split("\n")
        paralist, tag_para_list = tagmaker(plag_paragraphsplit, plagparanumber)
        vectorizer = TfidfVectorizer()
        tfIdfMat = vectorizer.fit_transform(tag_para_list)
        ##print(tfIdfMat.shape)
        #print(vectorizer.vocabulary_)
        feature_names = sorted(vectorizer.get_feature_names())
        #print(k)
        skDocsTfIdfdf = pd.DataFrame(tfIdfMat.todense(), index=(paralist), columns=feature_names)
        ##print(skDocsTfIdfdf)
        csim = cosine_similarity(plagtfidfMat,tfIdfMat)
        csimsk = pd.DataFrame(csim,index = (plagparalist), columns = (paralist))
        print(csimsk)


        #print(paranumber)

    #print(doc[0])
    # paranumber = doc[1].count("\n")
    # print(paranumber)