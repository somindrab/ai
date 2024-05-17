import pyprind
import pandas as pd
import numpy as np
import sys
import os

# example path to a file on this laptop is Datasets/aclImdb/train/pos/file.txt

basedir = 'Datasets/aclImdb'

labels = {'pos':1, 'neg':0}

pbar = pyprind.ProgBar(50000, stream=sys.stdout)

df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos','neg'):
        path = os.path.join(basedir, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),
                      'r',
                      encoding='utf-8') as infile:
                txt = infile.read()
                # no append in our version of pandas
                #df = df.append([[txt,labels[l]]],
                #            ignore_index=True)

                #so we use concat
                #note that this is different from the book -
                #we need this to get the right shape of the dataframe
                #i.e., one row with 2 columns being appended each time
                newdf = pd.DataFrame(np.array([[txt, labels[l]]]))
                #print(newdf.shape)
                df = pd.concat([df,newdf],
                               ignore_index=True)
                #print(df)
                pbar.update()

df.columns = ['review','sentiment']

#print(df.head)

#Note how this is all very 'structured' - the review files are in
#train and test, and then in pos and neg subdirectories.
#we want to shuffle all this up so that we have a dataset that is
#randomized, and from where we can slice a piece out for training and a piece
#for testing

np.random.seed(0)
#randomly permute our dataframe, and reindex it
df=df.reindex(np.random.permutation(df.index))
df.to_csv('Datasets/aclImdb/movie_data.csv',index=False,encoding='utf-8')
