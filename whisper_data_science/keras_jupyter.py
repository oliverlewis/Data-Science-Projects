
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import LSTM
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

seed = 7
np.random.seed(seed)


# In[2]:

nltk.download('stopwords')
stop = set(stopwords.words('english'))
porter = PorterStemmer()


# In[3]:

def clean_sentence(sentence):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(D|P)', sentence)
    sentence = re.sub('[\W]+', ' ', sentence.lower()) + ''.join(emoticons).replace('-', '')
    sentence = re.sub(r'(\d+)([a-z+]+)', r'n_age \2', sentence)
    sentence = re.sub(r'([a-z+]+)(\d+)', r'\1 n_age', sentence)
    sentence = re.sub(r'(\d+)', r'n_age', sentence)
    slang_words = {' u ': ' you ', ' ur ': ' your ', ' ru ': ' are you ',
                   ' r ': ' are ', ' k ': ' okay ', ' ok ': ' okay ', ' ya ': ' yes ',
                   ' wd ': ' with ', ' hv ': ' have ', ' gv ': ' give ', ' bf ': ' boyfriend ', ' gf ': ' girlfriend ',
                   ' lez ': ' lesbian ', ' les ': ' lesbian ', ' m ': ' male ', ' f ': ' female ',
                   ' wanna ': ' want to ', ' gonna ': ' going to ',
                   ' lesbo ': ' lesbian ', ' bc ': ' because ', ' plz ': ' please ', ' don t ': ' dont ',
                   ' can t ': ' cant ', ' won t ': ' wont '}
    for k, v in slang_words.items():
        sentence = sentence.replace(k, v)

    # Stemming words and removing stopwords:
    sentence = " ".join([word for word in sentence.split() if word not in stop])
    return sentence


# In[4]:

dataframe_train = pd.read_csv("data/train.csv")
dataset_train = dataframe_train.values
X_train = dataset_train[:, 1].astype(str)
X_train = map(clean_sentence, X_train)
Y_train = dataset_train[:, 0]


# In[5]:

dataframe_test = pd.read_csv("data/test.csv")
dataset_test = dataframe_test.values
X_test = dataset_test[:, 1].astype(str)
X_test = map(clean_sentence, X_test)
Y_test = dataset_test[:, 0]


# In[6]:

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
encoded_Y_test = encoder.transform(Y_test)


# In[7]:

y_train_cat = np_utils.to_categorical(encoded_Y_train)
y_test_cat = np_utils.to_categorical(encoded_Y_test)
print (y_train_cat.shape, y_test_cat.shape)


# In[8]:

nb_classes = np.max(encoded_Y_train)+1
print(nb_classes, 'classes')


# In[9]:

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_num = tokenizer.texts_to_sequences(X_train)
X_test_num = tokenizer.texts_to_sequences(X_test)
X_train_mat = tokenizer.sequences_to_matrix(X_train_num)
X_test_mat = tokenizer.sequences_to_matrix(X_test_num)


# In[10]:

print('X_train shape:', X_train_mat.shape)
print('X_test shape:', X_test_mat.shape)


# In[36]:

batch_size = 100
nb_epoch = 50


# In[42]:

# create model
def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_shape=(10249,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))   
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[43]:

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
estimator.fit(X_train_mat, y_train_cat)


# In[44]:

predictions = estimator.predict(X_test_mat)
print(set(predictions))
print(encoder.inverse_transform(predictions))


# In[45]:

print 'macro f1:', f1_score(encoded_Y_test, predictions, average='macro')


# In[ ]:



