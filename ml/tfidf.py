# using tfidf to construct a bag-of-words model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

#instantiate a count vectorizer
cv = CountVectorizer()

#our data
data = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet',
                 'and one and one is two'])

#create the bag
bag = cv.fit_transform(data)

#now let's apply tfidf to transform this bag to apply
#the tfidf relevancy transformation

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
np.set_printoptions(precision=2)

tfidf_bag = tfidf.fit_transform(bag)

print(tfidf_bag.toarray())

### So, the bag of words without relevancy is
# [[0 1 0 1 1 0 1 0 0]
#  [0 1 0 0 0 1 1 0 1]
#  [1 2 0 1 1 1 2 0 1]
#  [2 1 2 0 0 0 0 1 0]]


### and then with tfidf, with relevancy applied, it goes to
# [[0.   0.38 0.   0.57 0.57 0.   0.46 0.   0.  ]
#  [0.   0.38 0.   0.   0.   0.57 0.46 0.   0.57]
#  [0.33 0.43 0.   0.33 0.33 0.33 0.53 0.   0.33]
#  [0.57 0.19 0.72 0.   0.   0.   0.   0.36 0.  ]]


## the vocabulary is
#{'the': 6, 'sun': 4, 'is': 1, 'shining': 3, 'weather': 8, 'sweet': 5, 'and': 0, 'one': 2, 'two': 7}
