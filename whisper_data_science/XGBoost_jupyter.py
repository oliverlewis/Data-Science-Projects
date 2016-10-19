
# coding: utf-8

# In[1]:

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from xgboost import plot_importance, plot_tree
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import re
from csv import DictReader
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


# In[2]:

nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


# In[3]:

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
    for k, v in slang_words.items():
        sentence = sentence.replace(k, v)
    return sentence


# In[4]:

def read_data(name):
    text, targets = [], []

    with open('data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            sentence = item['text'].decode('utf8')
            sentence = clean_sentence(sentence)
            text.append(sentence)
            targets.append(item['category'])

    return text, targets


# In[5]:

def tokenizer_porter(sentence):
    return [porter.stem(word) for word in sentence.split()]


# In[6]:

text_train, targets_train = read_data('train')
text_test, targets_test = read_data('test')


# In[7]:

encoder = LabelEncoder()
encoder.fit(targets_train)
encoded_Y_train = encoder.transform(targets_train)
encoded_Y_test = encoder.transform(targets_test)


# In[8]:

eval_set = [(text_test, encoded_Y_test)]
model = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer_porter, stop_words=stop)),
    ('clf', XGBClassifier())
])


# In[21]:

n_estimators = [100, 150, 200]
max_depth = [4, 6, 8]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(clf__max_depth=max_depth, clf__n_estimators=n_estimators, clf__learning_rate=learning_rate)

kfold = StratifiedKFold(encoded_Y_train, n_folds=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold, verbose=1)


# In[ ]:

result = grid_search.fit(text_train, encoded_Y_train)


# Fitting 10 folds for each of 45 candidates, totalling 450 fits
# [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed: 47.2min
# [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed: 273.3min
# [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 631.1min
# [Parallel(n_jobs=-1)]: Done 450 out of 450 | elapsed: 646.4min finished

# In[ ]:

print 'Best parameter set: %s' % result.best_params_
print 'CV accuracy %0.3f' % result.best_score_
clf = result.best_estimator_


# Best parameter set: {'clf__max_depth': 8, 'clf__learning_rate': 0.1, 'clf__n_estimators': 200}
# CV accuracy -1.303

# In[ ]:

predictions = clf.predict(text_test)
print 'macro f1:', f1_score(encoded_Y_test, predictions, average='macro')


# macro f1: 0.46528675179
