import nltk
import re
import pandas as pd
import numpy as np
import itertools as it
import textstat
from scipy.sparse import coo_matrix, hstack
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option("display.max_rows", None, "display.max_columns", None)
documents = {'originaldoc_1': [1], 'originaldoc_2': [2], 'originaldoc_3': [3], 'originaldoc_4': [4], 'originaldoc_5': [5]
         }
input_document = "altered3_doc1"
def paragraphs (file, seperator = '/n'):
    #iterate a fileobj by paragraph
    lines = []
    for line in file:
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
def remove_numbers(text):
    text_nonum = re.sub(r'\d+', '', text)
    return text_nonum

def tagmaker(splitdocument,paragraphnum):
    tempnum = paragraphnum
    col = []
    paragraphnumrow = []
    sentencenumdata = []
    wordcount = []

    paralist = []
    l = 0
    document_taglist = []
    while l<= tempnum:
        col.append(0)
        paragraphnumrow.append(l)
        sentencenumdata.append(textstat.sentence_count(splitdocument[l]))
        wordcount.append((len(splitdocument[l]))/(textstat.sentence_count(splitdocument[l])))
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
            if j == 'NNS':
                tagstring = tagstring + j + " "
            elif j == 'NN':
                tagstring = tagstring + j + " "
            elif j == 'NNP':
                tagstring = tagstring + j + " "
            elif j == 'JJ':
                tagstring = tagstring + j + " "
            elif j == 'VB':
                tagstring = tagstring + j + " "
            elif j == 'VBD':
                tagstring = tagstring + j + " "
        document_taglist.append(tagstring)
        #print(tagstring)
    npcol = np.array(col)
    npparagraphrow = np.array(paragraphnumrow)
    npsentencedata = np.array(sentencenumdata)
    npwordData = np.array(wordcount)
    return paralist,document_taglist,npcol,npparagraphrow,npsentencedata,npwordData

for files,authornumber in documents.items():#for each file and authornumber in the dictionary split up as such
    docs = reading_in_files(files) #chunk file into paragraphs
    inputDoc = reading_in_files(input_document) #chunk file into paragraphs
    print("Document in question: " + input_document)
    #displays the "input" document being assessed
    print("Document being compared against: " + files)
    #displays the document that the "input" document is being checked against
    for doc in inputDoc[0]:#going through the input document
        inputParanumber = doc.count("\n")#counts the number of paragraphs
        input_paragraphsplit = doc.split("\n")#splits the document into paragraphs
        inputparalist, tag_input_list,npinputcol,npinputpara,npinputsentdata,npinputwordcount = tagmaker(input_paragraphsplit, inputParanumber)
        #retrieves the paragraph labels and tags
        input_vectorizer = TfidfVectorizer()#creates a tfidf vectorizer
        inputTfidf = input_vectorizer.fit_transform(tag_input_list)#fits the vectorizer to the tags
        inputsentencematrix = coo_matrix((npinputsentdata,(npinputpara,npinputcol)), shape=((inputParanumber+1),1))
        inputwordmatrix = coo_matrix((npinputwordcount,(npinputpara,npinputcol)),shape=((inputParanumber+1),1))
        tempinputmatrix = hstack([inputTfidf,inputsentencematrix])
        inputmatrix = hstack([tempinputmatrix,inputwordmatrix])
    for i in docs[0]:#for each document in the docs list
        paranumber = i.count("\n") #counts the number of paragraphs
        paragraphsplit = i.split("\n")#splits the documents into paragraphsplit list
        paralist, tag_para_list,npdoccol, npparagraphnum, npsentdata, npdocwordcount = tagmaker(paragraphsplit, paranumber)#retrieves the paragraph labels and tags
        vectorizer = TfidfVectorizer() #creates a tfidf vectorizer
        docTfidf = vectorizer.fit_transform(tag_para_list) #fits  the vectorizer to the tags
        docsentencematrix = coo_matrix((npsentdata,(npparagraphnum,npdoccol)), shape=((paranumber+1),1))
        docwordmatrix = coo_matrix((npdocwordcount,(npparagraphnum, npdoccol)),shape=((paranumber+1),1))
        tempdocmatrix = hstack([docTfidf,docsentencematrix])
        docmatrix = hstack([tempdocmatrix,docwordmatrix])
        csim = cosine_similarity(inputmatrix, docmatrix)
        #calculates the cosine similarity between input document and document from corpus
        csimdataframe = pd.DataFrame(csim,index = (inputparalist), columns = (paralist))
        #creates pandas dataframe to display cosine similarity
        percentagesum = 0 #variable to sum max similarities
        for rowIter in range(0,len(csim)):#from 0 to the number of paragraphs in input document
            values = csim[rowIter] #sets the similarities for the paragraph in index rowIter to values
            highvalue = max(values)#calculates the highest similarity between the paragraphs
            percentagesum = percentagesum + highvalue #adds the highest similarity to the percentage sum
            for colIter in range(0,len(values)):#from 0 to the number of paragraphs in corpus document
                if values[colIter] >= 0.99:
                    print(" ")
                    output_result = str(inputparalist[rowIter] + " in the assessed document has a strong match to "+
                                        paralist[colIter] + " in the original document " + files)
                    #displays paragraphs that are similar to one another
                    print(output_result)
                    print(" ")
                    print(inputparalist[rowIter] + " of assessed doc :\n" + input_paragraphsplit[rowIter])
                    print(paralist[colIter] + " of " + files + " :\n" + paragraphsplit[colIter])
                    print(" ")
        print("Maximum similarity is :" + (str((percentagesum/len(csim))*100)))#displays the average similarity
        print(" ")
        print(csimdataframe)#displays the cosine similarity matrix
        print("***********")


