
# coding: utf-8

# In[ ]:

from csv import DictReader
import pandas as pd
import re
from sklearn.metrics import confusion_matrix
import operator
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:

nltk.download('stopwords')
stop = stopwords.words('english')


# In[ ]:

porter = PorterStemmer()


# In[ ]:

def tokenizer(sentence):
    return sentence.split()


# In[ ]:

def tokenizer_porter(sentence):
    return [porter.stem(word) for word in sentence.split()]


# In[ ]:

def clean_sentence(sentence):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(D|P)', sentence)
    sentence = re.sub('[\W]+', ' ', sentence.lower()) + ''.join(emoticons).replace('-', '')
    sentence = re.sub(r'(\d+)([a-z+]+)', r'\1 \2', sentence)
    sentence = re.sub(r'([a-z+]+)(\d+)', r'\1 \2', sentence)
    slang_words = {' u ': ' you ', ' ur ': ' your ', ' ru ': ' are you ', 
             ' r ': ' are ', ' k ': ' okay ', ' ok ': ' okay ', ' ya ': ' yes ', 
             ' wd ': ' with ', ' hv ': ' have ', ' gv ': ' give ', ' bf ': ' boyfriend ', ' gf ': ' girlfriend ',
             ' lez ': ' lesbian ', ' les ': ' lesbian ', ' m ': ' male ', ' f ': ' female ',
             ' wanna ': ' want to ', ' gonna ': ' going to ',
             ' lesbo ': ' lesbian ', ' bc ': ' because ', ' plz ': ' please ', ' don t ': ' dont ',
             ' can t ': ' cant ', ' won t ': ' wont '}
    for k,v in slang_words.items():
        sentence = sentence.replace(k, v)
    return sentence


# In[ ]:

def read_data(name):
    text, targets = [], []

    with open('data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            sentence = item['text'].decode('utf8')
            sentence = clean_sentence(sentence)
            text.append(sentence)
            targets.append(item['category'])

    return text, targets


# In[ ]:

text_train, targets_train = read_data('train')
text_test, targets_test = read_data('test')


# In[ ]:

encoder = LabelEncoder()
encoder.fit(targets_train)
encoded_Y_train = encoder.transform(targets_train)
encoded_Y_test = encoder.transform(targets_test)


# In[ ]:

tfidf = TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       preprocessor=None)


# In[ ]:

param_grid = [{
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [None, stop],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [1.0, 10.0, 100.0]
            },
             {
        'vect__ngram_range': [(1,1), (1,2)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [1.0, 10.0, 100.0]
            }]


# In[ ]:

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])


# In[ ]:

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)


# Fitting 10 folds for each of 72 candidates, totalling 720 fits
# [Parallel(n_jobs=-1)]: Done  18 tasks      | elapsed:    5.3s
# [Parallel(n_jobs=-1)]: Done 168 tasks      | elapsed: 10.0min
# [Parallel(n_jobs=-1)]: Done 418 tasks      | elapsed: 13.3min
# [Parallel(n_jobs=-1)]: Done 720 out of 720 | elapsed: 22.6min finished

# In[ ]:

gs_lr_tfidf.fit(text_train, encoded_Y_train)


# In[ ]:

print 'Best parameter set: %s' % gs_lr_tfidf.best_params_


# Best parameter set: {'vect__ngram_range': (1, 2), 'vect__tokenizer': <function tokenizer_porter at 0x7f257fc60140>, 'vect__stop_words': None, 'vect__norm': None, 'clf__penalty': 'l1', 'clf__C': 1.0, 'vect__use_idf': False}

# In[ ]:

print 'CV accuracy %0.3f' % gs_lr_tfidf.best_score_


# CV accuracy 0.616

# In[ ]:

clf = gs_lr_tfidf.best_estimator_


# In[ ]:

prediction = clf.predict(text_test)
print 'macro f1:', f1_score(encoded_Y_test, prediction, average='macro')


# macro f1: 0.532216461866

# In[ ]:

orig_predictions = encoder.inverse_transform(prediction)
print list(encoder.classes_)
pd.DataFrame(confusion_matrix(targets_test, orig_predictions)).to_csv('confusion_mat.csv')

