import nltk
import re
import string
import sklearn
import pandas as pd
import itertools as it
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
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


for files,authornumber in documents.items():
    docs = reading_in_files(files)
    plagdoc = reading_in_files(plagiairised_document)
    for doc in plagdoc[0]:
        plagparanumber = doc.count("\n")
        plag_paragraphsplit = doc.split("\n")
        tempnum = plagparanumber
        plagparalist = []
        pl = 0
        tag_plag_list = []
        while pl<=tempnum:
            plaglabel = "paragraph" + str(pl+1)
            plagparalist.append(plaglabel)
            pl = pl+1
        for val in plag_paragraphsplit:
            remove_numbers(val)
            plag_tokens = nltk.word_tokenize(val)
            plagtagging = nltk.pos_tag(plag_tokens)
            plagwords,ptaglist = zip(*plagtagging)
            pls = ""
            for pj in ptaglist:
                if pj == 'NN':
                    pls = pls + pj + " "
                elif pj == 'NNS':
                    pls = pls + pj + " "
            tag_plag_list.append(pls)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        plag_vectorizer = TfidfVectorizer()
        plagtfidfMat = plag_vectorizer.fit_transform(tag_plag_list)
        plag_featurenames = sorted(plag_vectorizer.get_feature_names())
        plagSk = pd.DataFrame(plagtfidfMat.todense(), index=(plagparalist), columns=plag_featurenames)

    for i in docs[0]:
        #print(i)
        paranumber = i.count("\n")
        paragraphsplit = i.split("\n")
        k = paranumber
        paralist = []
        p = 0
        tag_para_list = []
        while p <= k:
            paralabel = "paragraph" + str(p+1)
            paralist.append(paralabel)
            p = p + 1
        for x in paragraphsplit:
            remove_numbers(x)
            tokens = nltk.word_tokenize(x)
            postagging = nltk.pos_tag(tokens)
            words,taglist = zip(*postagging)
            #print(x)
            s = ""
            for j in taglist:
                if j == 'NN':
                    s = s + j + " "
                elif j == 'NNS':
                    s = s + j + " "
            tag_para_list.append(s)
            #print(taglist)
        #print(tag_para_list)
        vectorizer = TfidfVectorizer()
        tfIdfMat = vectorizer.fit_transform(tag_para_list)
        ##print(tfIdfMat.shape)
        #print(vectorizer.vocabulary_)
        feature_names = sorted(vectorizer.get_feature_names())
        #print(k)
        skDocsTfIdfdf = pd.DataFrame(tfIdfMat.todense(), index=(paralist), columns=feature_names)
        ##print(skDocsTfIdfdf)
        csim = cosine_similarity(plagtfidfMat,tfIdfMat)
        csimsk = pd.DataFrame(csim)
        print(csimsk)


        #print(paranumber)

    #print(doc[0])
    # paranumber = doc[1].count("\n")
    # print(paranumber)