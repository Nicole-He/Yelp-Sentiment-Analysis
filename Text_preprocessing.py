#Load Libraries
import numpy as np
import pandas as pd
#Packages to process the comments
###corpora.dictionary: mapping between words and their integer ids
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string

###Step1: Create stopword list
stop_words = set(stopwords.words('english'))
######Update stopwords by adding punctuations using string.punctuation
stop_words.update(string.punctuation)
print(stop_words)

###Step2: Create stemmer list (https://en.wikipedia.org/wiki/Stemming: example section)
stemmer = SnowballStemmer('english')

###Step3: Preprocess training data
print("pre-processing raw training doc ...")
processed_doc_train = []

for doc in raw_doc_train:
    #seperate text into words only
    tokens = word_tokenize(doc)
    #remove stopwords
    filtered = [word for word in tokens if word not in stop_words]
    #get stem words
    stemmed = [stemmer.stem(word) for word in filtered]
    #save processed words
    processed_doc_train.append(stemmed)

###Step4: Preprocess testing data
print("pre-processing raw testing doc ...")
processed_doc_test = []

for doc in raw_doc_test:
    #seperate text into words only
    tokens = word_tokenize(doc)
    #remove stopwords
    filtered = [word for word in tokens if word not in stop_words]
    #get stem words
    stemmed = [stemmer.stem(word) for word in filtered]
    #save processed words
    processed_doc_test.append(stemmed)

###Step5: Combine all the key words in training and testing data
processed_doc_all = np.concatenate((processed_doc_train, processed_doc_test), axis = 0)

###Step6: Create a dictionary containing all the key words
dictionary = corpora.Dictionary(processed_doc_all)
dictionary_size = len(dictionary.keys())
print("dictionary size: {0}".format(dictionary_size))
######List of words and their ids
#for i in dictionary:
    #print(i, dictionary[i])
######Save the dictionary
#dictionary.save('dictionary.dict')


###Step7: Map words to numbers
print("converting to token ids ...")
word_id_train, word_id_len = [], []

######Map training texts to numbers
for doc in processed_doc_train:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_train.append(word_ids)
    word_id_len.append(len(word_ids))

word_id_test, word_ids = [], []
######Map training texts to numbers
for doc in processed_doc_test:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_test.append(word_ids)
    word_id_len.append(len(word_ids))

###Step8: Make all inputs the same length
######Plot the distribution of text length
plt.hist(word_id_len, bins = 100)
######Find mean, std and max
pd.DataFrame(word_id_len).describe() #mean:78, std:85, max:514
######Determine the maximum length
seq_len = np.round((np.mean(word_id_len) + 3*np.std(word_id_len))).astype(int)
######Pad sequences of features/predictors to have uniform len=seq_len either truncating or padding zeros
word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen = seq_len)
word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen = seq_len)

###Step9: Convert sentiment to one-hot vectors
y_train_enc = np_utils.to_categorical(sentiment_train-1, num_labels)
y_test_enc = np_utils.to_categorical(sentiment_test-1, num_labels)

#Save processed data
pd.DataFrame(word_id_train).to_csv('word_id_train.csv', index=False)
pd.DataFrame(word_id_test).to_csv('word_id_test.csv', index=False)
pd.DataFrame(y_train_enc).to_csv('y_train_enc.csv', index=False)
pd.DataFrame(y_test_enc).to_csv('y_test_enc.csv', index=False)
#dictionary_size = 248823
#num_labels = 5
