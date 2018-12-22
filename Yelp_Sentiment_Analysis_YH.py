#Sentiment Analysis on Yelp Comments

#Data Description: You are given a data set yelp.csv containing two features:
#1. stars: users rate the business from1-5.
#2. text: users’ reviews

#Section 1: Load Libraries
import numpy as np
import pandas as pd
#Packages to process the comments
###corpora.dictionary: mapping between words and their integer ids
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
#RNN with LSTM to handle sentiment
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
#To split dataset into training data and testing data
from sklearn.model_selection import train_test_split
#To plot results
from matplotlib import pyplot as plt
import seaborn as sns

#Section 2: Data Preprocessing
#To get same random number
np.random.seed(0)

#Load data from local direction
df = pd.read_csv('C:/Users/Yiling/Desktop/DSCI 619 Deep Learning/HW/Project 2/yelp.csv', header = 0)

#Structure of dataset
df.columns #['stars', 'text']
df.shape #(500000, 2)

###To create a small sample dataset for debugging
#df = df.loc[0:1000, :]

#Reorganize the dataset
df['text length'] = df['text'].apply(len)
raw_doc = df['text'].values #including all comments
sentiment = df['stars'].values #range from 1 to 5
num_labels = len(np.unique(sentiment))

#Change matplotlib graph style to ggplot
plt.style.use('ggplot')

#Plot data
###Sentiment
dist_sentiment = df.groupby(['stars']).size()
plt.bar(np.unique(sentiment), dist_sentiment, color = 'b')
plt.xlabel("Stars")
plt.ylabel("Frequency")
plt.title("Yelp Sentiment")
plt.show()

#Split dataset into training data and testing data, split ratio is 90/100
raw_doc_train, raw_doc_test, sentiment_train, sentiment_test = train_test_split(raw_doc, sentiment, test_size=0.1, random_state = 0)
###Double check dataset dimensions
raw_doc_train.shape #(450000,)
raw_doc_test.shape #(50000,)
#Save data for traditional machine learning methods
#pd.DataFrame(sentiment_train).to_csv('sentiment_train.csv', index = False)
#pd.DataFrame(sentiment_test).to_csv('sentiment_test.csv', index = False)


#Test pre-processing
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

#Section 3: Build Models

################
###LSTM Model###
################
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
from keras.layers import Embedding, Dropout, CuDNNLSTM, Dense
#Build LSTM Model
print("fitting LSTM ...")
model = Sequential()
model.add(Embedding(dictionary_size, 128))
model.add(Dropout(0.5))
model.add(CuDNNLSTM(64))
model.add(Dense(num_labels, activation='softmax'))

#Review model structure
model.summary()

#Save model after every epoch
checkpoint = ModelCheckpoint('Yelp_Sentiment_Analysis.h5', #model filename
                             monitor = 'val_loss', #quantity to monitor
                             verbose = 0, #verbosity - 0 or 1
                             save_best_only = True, #the latest best model will not be overwitten
                             mode = 'auto') #The decision to overwirte model is made automatically depending on the quantity to monitor.

earlystopper = EarlyStopping(monitor='val_loss', patience=0, mode='auto')

#Configure the model for training
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 1.0e-4),
              metrics = ['accuracy'])

#Fit the model
model_details = model.fit(word_id_train, y_train_enc, 
                          nb_epoch=10, batch_size=256,
                          validation_data = (word_id_test, y_test_enc),
                          callbacks = [checkpoint, earlystopper],
                          verbose = 1)

########################
###LSTM Model for GPU###
########################
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
from keras.layers import Embedding, Dropout, CuDNNLSTM, Dense
#Build LSTM Model
print("fitting LSTM for GPU...")
model = Sequential()
model.add(Embedding(dictionary_size, 64))
model.add(Dropout(0.5))
model.add(CuDNNLSTM(64, return_sequences=True)) 
model.add(Dropout(0.5))
model.add(CuDNNLSTM(64)) 
model.add(Dense(num_labels, activation='softmax'))

#Review model structure
model.summary()

#Save model after every epoch
checkpoint = ModelCheckpoint('Yelp_Sentiment_Analysis.h5', #model filename
                             monitor = 'val_loss', #quantity to monitor
                             verbose = 0, #verbosity - 0 or 1
                             save_best_only = True, #the latest best model will not be overwitten
                             mode = 'auto')#The decision to overwirte model is made automatically depending on the quantity to monitor.

earlystopper = EarlyStopping(monitor='val_loss', patience=0, mode='auto')

#Configure the model for training
#optimizer = Adam(lr=1.0E-4)
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#Fit the model
model_details = model.fit(word_id_train, y_train_enc, 
                          nb_epoch=10, batch_size=64,
                          validation_data = (word_id_test, y_test_enc),
                          callbacks = [checkpoint, earlystopper],
                          verbose = 1)

######################
###CNN + LSTM Model###
######################
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Dense, CuDNNLSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
from keras.models import Sequential

#Build LSTM Model
print("fitting CNN + LSTM ...")
model = Sequential()
model.add(Embedding(dictionary_size, 64))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, padding = 'same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.5))
model.add(CuDNNLSTM(100))
model.add(Dense(num_labels, activation='softmax'))

#Review model structure
model.summary()

#Save model after every epoch
checkpoint = ModelCheckpoint('Yelp_Sentiment_Analysis_CNNLSTM.h5', #model filename
                             monitor = 'val_loss', #quantity to monitor
                             verbose = 0, #verbosity - 0 or 1
                             save_best_only = True, #the latest best model will not be overwitten
                             mode = 'auto') #The decision to overwirte model is made automatically depending on the quantity to monitor.

earlystopper = EarlyStopping(monitor='val_loss', patience=0, mode='auto')

#Configure the model for training
optimizer = Adam(lr=1.0E-4)
model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])

#Fit the model
model_details = model.fit(word_id_train, y_train_enc, 
                          nb_epoch=10, batch_size=64,
                          validation_data = (word_id_test, y_test_enc),
                          callbacks = [checkpoint, earlystopper],
                          verbose = 1)


#Section 4: Results
#Plot the accuracy and loss
###Create sub-plots
fig, axs = plt.subplots(1,2,figsize=(15,5))
    
###Summarize history for accuracy
axs[0].plot(range(1,len(model_details.history['acc'])+1),model_details.history['acc'])
axs[0].plot(range(1,len(model_details.history['val_acc'])+1),model_details.history['val_acc'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_xticks(np.arange(1,len(model_details.history['acc'])+1),len(model_details.history['acc'])/10)
axs[0].legend(['train', 'val'], loc='best')
    
###Summarize history for loss
axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
axs[1].legend(['train', 'val'], loc='best')
   
###Show the plot
plt.show()
###Save the plots
plt.savefig('Yelp_Sentiment_Analysis_Results.png')

#Prediction
test_pred = model.predict_classes(word_id_test)

#Evaluate the model
scores = model.evaluate(word_id_test, y_test_enc, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1]*100))







