#Importing the required libraries
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict 
 
#defining dataset
data = ['There was a child','The child had a ball','The child played with the ball']
 
#tokenizing text.
sentences = []
vocab = []
for sent in data:
    x = word_tokenize(sent)
    sentence = [w.lower() for w in x if w.isalpha() ]
    sentences.append(sentence)
    for word in sentence:
        if word not in vocab:
            vocab.append(word)
 
#word frequency
len_vector = len(vocab)

#assigning an index to each word in vocabulary
index_word = {}
i = 0
for word in vocab:
    index_word[word] = i 
    i += 1

#creating BoW model
def bag_of_words(sent):
    count_dict = defaultdict(int)
    vec = np.zeros(len_vector)
    for item in sent:
        count_dict[item] += 1
    for key,item in count_dict.items():
        vec[index_word[key]] = item
    return vec

vector = bag_of_words(sentences[0])
print("Bag of Words: ",vector)