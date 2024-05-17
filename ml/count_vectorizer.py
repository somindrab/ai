# CountVectorizer constructs the bag of words model. See paper notes.

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
data = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet',
                 'and one and one is two'])

bag = cv.fit_transform(data)

print(cv.vocabulary_)

##print the bag of words

print(bag.toarray())

### Output is
#(pyml) som@som-linux-laptop:~/prog/pyml$ python count_vectorizer.py 
#{'the': 6, 'sun': 4, 'is': 1, 'shining': 3, 'weather': 8, 'sweet': 5, 'and': 0, 'one': 2, 'two': 7}

# [[0 1 0 1 1 0 1 0 0]
#  [0 1 0 0 0 1 1 0 1]
#  [1 2 0 1 1 1 2 0 1]
#  [2 1 2 0 0 0 0 1 0]]
