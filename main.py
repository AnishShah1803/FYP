import nltk
import re
import pandas as pd
import numpy as np
import itertools as it
import textstat
from scipy.sparse import coo_matrix, hstack
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option("display.max_rows", None, "display.max_columns", None)
class tf:
    header = "\033[95m"
    underline = "\033[4m"
    bold = "\033[1m"
    bc = "\033[0m"
    blue = "\033[94m"
    green = "\033[92m"
    red = "\033[91m"
documents = {'originaldoc_1': [1], 'originaldoc_2': [2], 'originaldoc_3': [3], 'originaldoc_4': [4], 'originaldoc_5': [5]
         }
#input_document = "altered3_doc1"
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
            elif j == 'VBP':
                tagstring = tagstring + j + " "
        document_taglist.append(tagstring)
        #print(tagstring)

    return paralist,document_taglist

def main():
    document_check = 0
    while True:
        try:
            document_check = int(input("Enter the document number you would liketo check between 1 and 5: "))
        except ValueError:
            print("You need to enter a number")
            continue
        else:
            if 1<= document_check <= 5:
                pass
            else:
                main()
            break
    input_alteration = 0
    while True:
        try:
            input_alteration = int(input("Enter the alteration number you would like to check between 1 and 3: "))
        except ValueError:
            print("You need to enter a number")
            continue
        else:
            if 1<= input_alteration <= 3:
                pass
            else:
                main()
            break

    input_document = ("altered"+str(input_alteration)+"_doc"+str(document_check))
    all_similarities = []
    for files,authornumber in documents.items():#for each file and authornumber in the dictionary split up as such
        docs = reading_in_files(files) #chunk file into paragraphs
        inputDoc = reading_in_files(input_document) #chunk file into paragraphs
        print("**********************************")
        print(tf.header + "Document in question: " + input_document + tf.bc)
        #displays the "input" document being assessed
        print(tf.header + "Document being compared against: " + files + tf.bc)
        #displays the document that the "input" document is being checked against
        for doc in inputDoc[0]:#going through the input document
            inputParanumber = doc.count("\n")#counts the number of paragraphs
            input_paragraphsplit = doc.split("\n")#splits the document into paragraphs
            inputparalist, tag_input_list = tagmaker(input_paragraphsplit, inputParanumber)
            #retrieves the paragraph labels and tags
            input_vectorizer = CountVectorizer()#creates a tfidf vectorizer
            inputTfidf = input_vectorizer.fit_transform(tag_input_list)#fits the vectorizer to the tags
        for i in docs[0]:#for each document in the docs list
            paranumber = i.count("\n") #counts the number of paragraphs
            paragraphsplit = i.split("\n")#splits the documents into paragraphsplit list
            paralist, tag_para_list = tagmaker(paragraphsplit, paranumber)#retrieves the paragraph labels and tags
            vectorizer = CountVectorizer() #creates a tfidf vectorizer
            docTfidf = vectorizer.fit_transform(tag_para_list) #fits  the vectorizer to the tags
            csim = cosine_similarity(inputTfidf, docTfidf)
            #calculates the cosine similarity between input document and document from corpus
            csimdataframe = pd.DataFrame(csim,index = (inputparalist), columns = (paralist))
            #creates pandas dataframe to display cosine similarity
            percentagesum = 0 #variable to sum max similarities
            sumcounter = 0
            #percentagesum = 0
            for rowIter in range(0,len(csim)):#from 0 to the number of paragraphs in input document
                values = csim[rowIter] #sets the similarities for the paragraph in index rowIter to values

                for colIter in range(0,len(values)):#from 0 to the number of paragraphs in corpus document
                    inputsentencecount = textstat.sentence_count(input_paragraphsplit[rowIter])
                    docsentencecount = textstat.sentence_count(paragraphsplit[colIter])
                    if (((input_paragraphsplit[rowIter].count(" ")) * 0.80) < (paragraphsplit[colIter].count(" ")) < ((input_paragraphsplit[rowIter].count(" ")) * 1.20)) or (((paragraphsplit[colIter].count(" ")) * 0.80) < (input_paragraphsplit[rowIter].count(" ")) < ((paragraphsplit[colIter].count(" ")) * 1.20)):
                        if (((inputsentencecount)*0.80) < (docsentencecount) < ((inputsentencecount)*1.20)) or (((docsentencecount)*0.80) <(inputsentencecount) <((docsentencecount)*1.20)):

                            if values[colIter] >= 0.90:
                                print(" ")
                                percentagesum = percentagesum + values[colIter]
                                sumcounter = sumcounter + 1
                                output_result = str(tf.blue + inputparalist[rowIter] + " in the assessed document has a strong match to "+
                                        paralist[colIter] + " in the original document " + files + tf.bc)
                                #displays paragraphs that are similar to one another
                                print(output_result)
                                #print(" ")
                                print(tf.bold + tf.underline + inputparalist[rowIter] + " of assessed doc :\n" + tf.bc + input_paragraphsplit[rowIter])
                                print(tf.bold + tf.underline + paralist[colIter] + " of " + files + " :\n" + tf.bc + paragraphsplit[colIter])
                                print(" ")
                                print("The similarity percentage between the two is : " + tf.underline +  str("{:.2f}".format(values[colIter]*100)) + "%" + tf.bc)
                                break
                            break

            if sumcounter <=1:
                print(tf.red + "There is negligable similarity between the documents" + tf.bc)
                all_similarities.append(0)
            else:
                percentage = ((percentagesum/sumcounter)*100)
                print("Similarity between documents is : " + tf.red + str("{:.2f}".format(percentage)) + "%" + tf.bc)
                all_similarities.append(percentage)
            print(" ")
                #print(csimdataframe)#displays the cosine similarity matrix
    finalsimilar = all_similarities.index(max(all_similarities))
    print(tf.green + "The input document " + str(input_document) + " is most similar to originaldoc_" + str(finalsimilar+1) + " with a similarity of " + str("{:.2f}".format(max(all_similarities))) + tf.bc)
main()