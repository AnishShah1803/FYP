import nltk
import re
import string
import sklearn
import pandas as pd
import itertools as it
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
documents = {'Author1': [1], 'Author2': [2], 'Author3': [3], 'Author4': [4], 'Author5': [5]
         }
#plagiairised document = ""
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
    for filename in filenames:
        with open(f'data/originaldoc_{filename}.txt') as f:
            paras = paragraphs(f)
            for para, group in zip(paras, it.cycle(paragraph_lists)):
                group.append(para)

    #print(paragraph_lists)
    return paragraph_lists
    # return '\n'.join(strings)
def remove_numbers(text):
    text_nonum = re.sub(r'\d+', '', text)
    return text_nonum

# for author,files in documents.items():
#     text_by_author[author] = reading_in_files(files)
for author,files in documents.items():
    docs = reading_in_files(files)
    for i in docs[0]:
        #print(i)
        paranumber = i.count("\n")
        paragraphlist = i.split("\n")
        k = paranumber
        paralist = []
        p = 0
        tag_para_list = []
        while p <= k:
            paralabel = "paragraph" + str(p+1)
            paralist.append(paralabel)
            p = p + 1
        for x in paragraphlist:
            remove_numbers(x)
            tokens = nltk.word_tokenize(x)
            postagging = nltk.pos_tag(tokens)
            words,taglist = zip(*postagging)
            #print(x)
            s = ""
            for j in taglist:
                s = s + j + " "
            tag_para_list.append(s)
            #print(taglist)
        #print(tag_para_list)
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer()
        tfIdfMat = vectorizer.fit_transform(tag_para_list)
        print(tfIdfMat.shape)
        #print(vectorizer.vocabulary_)
        feature_names = sorted(vectorizer.get_feature_names())
        #print(k)
        skDocsTfIdfdf = pd.DataFrame(tfIdfMat.todense(), index=(paralist), columns=feature_names)
        print(skDocsTfIdfdf)



        #print(paranumber)

    #print(doc[0])
    # paranumber = doc[1].count("\n")
    # print(paranumber)










