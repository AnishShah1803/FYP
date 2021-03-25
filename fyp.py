import nltk #natural language toolkit
import re #regex
import pandas as pd #pandas package
import itertools as it #itertools package
import textstat #textstat package
from sklearn.feature_extraction.text import CountVectorizer #importing CountVectorizer function
from sklearn.metrics.pairwise import cosine_similarity #importing cosine similarity function
pd.set_option("display.max_rows", None, "display.max_columns", None) #makes sure full dataframes are shown
class tf: #textformatting
    header = "\033[95m"
    underline = "\033[4m"
    bold = "\033[1m"
    bc = "\033[0m"
    blue = "\033[94m"
    green = "\033[92m"
    red = "\033[91m"
documents = {'originaldoc_1': [1], 'originaldoc_2': [2], 'originaldoc_3': [3], 'originaldoc_4': [4], 'originaldoc_5': [5]
         } #dictionary of original documents
def paragraphs (file, seperator = '\n'): #iterates through file by new lines and joins together to form string that has newline to replace gaps
    lines = [] #instantiate empty list to hold lines
    for line in file:#checks for sperator and joins to previous line
        if line == seperator and lines:
            yield ''.join(lines)
            lines = []
        else:
            lines.append(line)
    yield ''.join(lines)
def reading_in_files(filenames): #seperates the dictionary values into para and location
    paragraph_lists = [[]]#list of lists that holds
    with open(f'data/{filenames}.txt') as f:
        paras = paragraphs(f)
        for para, location in zip(paras, it.cycle(paragraph_lists)):
            location.append(para)
    return paragraph_lists
def remove_numbers(text):#replaces numbers in text with empty char
    text_nonum = re.sub(r'\d+', '', text)
    return text_nonum

def tagmaker(splitdocument,paragraphnum):#applies pos tags to input document and returns list of words per paragraph and list of tags per paragraph
    tempnum = paragraphnum
    paralist = []
    l = 0
    document_taglist = []
    while l<= tempnum:#from range 0 to the number of paragraphs
        plaglabel = "paragraph " + str(l+1)
        paralist.append(plaglabel)#list of labels for paragraphs to be able to name later on
        l = l+1 #increment through while loop
    for item in splitdocument: #per paragraph
        remove_numbers(item) #applies remove numbers to the paragraph
        tokens = nltk.word_tokenize(item) #tokenize the words and store them in tokens
        applytags = nltk.pos_tag(tokens) #apply pos tags to the tokens
        words,taglist = zip(*applytags) #seperate the tags and words into two seperate lists
        tagstring = " " #empty string to hold the tags at a later point
        for j in taglist:
            if j == 'NNS': #plural noun
                tagstring = tagstring + j + " "
            elif j == 'NN':#singular noun
                tagstring = tagstring + j + " "
            elif j == 'NNP':#proper noun
                tagstring = tagstring + j + " "
            elif j == 'JJ':#adjective
                tagstring = tagstring + j + " "
            elif j == 'VB':#verb in base form
                tagstring = tagstring + j + " "
            elif j == 'VBD': #past tense verb
                tagstring = tagstring + j + " "
            elif j == 'VBP': #present tense verb
                tagstring = tagstring + j + " "
        document_taglist.append(tagstring)#appends string of tags per paragraph to document_taglist
        #print(tagstring)

    return paralist,document_taglist #returns the labels for the paragraphs and the tags

def main():
    document_check = 0 #sets the initial value to zero before while loop
    input_alteration = 0 #sets initial value to zero before while loop
    while True: #continue to ask for input whilst entry is not a number between 1 and 5
        try:
            document_check = int(input("Enter the document number you would like to check between 1 and 5: "))
        except ValueError:
            print("You need to enter a number")
            continue
        else:
            if 1<= document_check <= 5:
                pass
            else:
                continue

            break

    while True: #continue to ask for input whilst entry is not a number between 1 and 3
        try:
            input_alteration = int(input("Enter the alteration number you would like to check between 1 and 3: "))
        except ValueError:
            print("You need to enter a number")
            continue
        else:
            if 1<= input_alteration <= 3:
                pass
            else:
                continue
            break

    input_document = ("altered"+str(input_alteration)+"_doc"+str(document_check)) #creating the name for the input document
    all_similarities = [] #list to hold the maximum similarities of each document where the index represents the document
    matches = [] #list to hold number of matches for each document where the index represents the document
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
            input_vectorizer = CountVectorizer()#creates a countvectorizer vectorizer
            inputCountVect = input_vectorizer.fit_transform(tag_input_list)#fits the vectorizer to the tags
            #inputsk = pd.DataFrame(inputCountVect.todense())
            #print(inputsk)
            #print(" ")
        for i in docs[0]:#for each document in the docs list
            paranumber = i.count("\n") #counts the number of paragraphs
            paragraphsplit = i.split("\n")#splits the documents into paragraphsplit list
            paralist, tag_para_list = tagmaker(paragraphsplit, paranumber)#retrieves the paragraph labels and tags
            vectorizer = CountVectorizer() #creates a tfidf vectorizer
            docCountVect = vectorizer.fit_transform(tag_para_list) #fits  the vectorizer to the tags
            #docsk = pd.DataFrame(docCountVect.todense())
            #print(docsk)
            csim = cosine_similarity(inputCountVect, docCountVect)
            #calculates the cosine similarity between input document and document from corpus
            csimdataframe = pd.DataFrame(csim,index = (inputparalist), columns = (paralist))
            #creates pandas dataframe to display cosine similarity
            percentagesum = 0 #variable to sum max similarities
            sumcounter = 0
            #percentagesum = 0
            for rowIter in range(0,len(csim)):#from 0 to the number of paragraphs in input document
                values = csim[rowIter] #sets the similarities for the paragraph in index rowIter to values

                for colIter in range(0,len(values)):#from 0 to the number of paragraphs in corpus document
                    inputsentencecount = textstat.sentence_count(input_paragraphsplit[rowIter]) #counting  the number of sentences in input document
                    docsentencecount = textstat.sentence_count(paragraphsplit[colIter]) #counting the number of sentences in original document
                    if (((input_paragraphsplit[rowIter].count(" ")) * 0.80) < (paragraphsplit[colIter].count(" ")) < ((input_paragraphsplit[rowIter].count(" ")) * 1.20)) or (((paragraphsplit[colIter].count(" ")) * 0.80) < (input_paragraphsplit[rowIter].count(" ")) < ((paragraphsplit[colIter].count(" ")) * 1.20)):
                        #checks the number of words per paragraph fits in a margin of 20%
                        if (((inputsentencecount)*0.90) < (docsentencecount) < ((inputsentencecount)*1.10)) or (((docsentencecount)*0.90) <(inputsentencecount) <((docsentencecount)*1.10)):
                            #checks the number of sentences fits within a margin of 10%
                            if values[colIter] >= 0.85: #checking similarity is above 0.85
                                print(" ")
                                percentagesum = percentagesum + values[colIter] #adds the similarity to the total similarity
                                sumcounter = sumcounter + 1 #increments the number of matches
                                output_result = str(tf.blue + inputparalist[rowIter] + " in the assessed document has a strong match to "+
                                        paralist[colIter] + " in the original document " + files + tf.bc)
                                #creates string to show what paragraphs match
                                print(output_result)#displays the match
                                print(tf.bold + tf.underline + inputparalist[rowIter] + " of assessed doc :\n" + tf.bc + input_paragraphsplit[rowIter]) #displays content of matching paragraph
                                print(tf.bold + tf.underline + paralist[colIter] + " of " + files + " :\n" + tf.bc + paragraphsplit[colIter]) #displays content of matching paragraph
                                print(" ")
                                print("The similarity percentage between the two is : " + tf.underline +  str("{:.2f}".format(values[colIter]*100)) + "%" + tf.bc) #displays the similarity between the documents to 2 dp
                                break
                            break
            matches.append(sumcounter)#adds the number of matches to the matches list
            if sumcounter ==0: #checks if there are no matches
                print(tf.red + "There are neglegable matches between the documents" + tf.bc)
                all_similarities.append(0) #appends no similarity
            elif sumcounter ==1: #checks for 1 similarity
                print("Similarity between documents is : " + tf.red + str("{:.2f}".format(percentagesum*100)) + "%" + tf.bc) #displays similarity percentage
                print("However there is just 1 match so it may not be necessary to investigate") #warning message
                all_similarities.append(percentagesum*100) #adds percentage to all_similarities
            else:#when sumcounter > 1
                percentage = ((percentagesum/sumcounter)*100) #calculates average percentage
                print("Similarity between documents is : " + tf.red + str("{:.2f}".format(percentage)) + "%" + tf.bc) #displays similarity percentage
                all_similarities.append(percentage) #adds percentage to all_similarities
            print(" ")
                #print(csimdataframe)#displays the cosine similarity matrix
    finalmatches = matches.index((max(matches)))#sets the final decision as the document with most matches
    print(tf.green + "The input document " + str(input_document) + " is most similar to originaldoc_" + str(finalmatches+1) + " with a similarity of " + str("{:.2f}".format(all_similarities[finalmatches])) + "%" + tf.bc)
    #displays overall what document is most similar
main()
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
#https://lzone.de/examples/Python%20re.sub
#https://pandas.pydata.org/pandas-docs/version/0.25.1/reference/api/pandas.DataFrame.to_dense.html
#https://pypi.org/project/textstat/#:~:text=textstat%20requires%20at%20least%203%20sentences%20for%20a%20result.
#https://www.geeksforgeeks.org/python-itertools-cycle/
#https://docs.python.org/3/library/itertools.html
#https://www.nltk.org/book/ch05.html

