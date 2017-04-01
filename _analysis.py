import spacy
import pandas as pd
import numpy as np
from collections import Counter
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Display plots in this notebook, instead of externally. 
# from pylab import rcParams
# rcParams['figure.figsize'] = 16, 8

## Load the models 
nlp = spacy.load('en')

grail_raw = unicode(open('grail.txt').read().decode('utf8'))
pride_raw = unicode(open('pride.txt').read().decode('utf8'))

# Parse the texts. These commands might take a little while. 
grail = nlp(grail_raw)
pride = nlp(pride_raw)

# print(pride[0])
# print(pride[:10])

# print(next(pride.sents))
# prideSents = list(pride.sents)
# print(prideSents[-1])


# prideSentenceLengths = [len(sent) for sent in prideSents]
# max_sent = [sent for sent in prideSents if len(sent) == max(prideSentenceLengths)]
# print(max_sent)


# props = [prop for prop in dir(pride[4]) if not prop.startswith('_')]
# print props


# print pride[4].i

def locations(needle, haystack): 
    """ 
    Make a list of locations, bin those into a histogram, 
    and finally put it into a Pandas Series object so that we
    can later make it into a DataFrame. 
    """
    return pd.Series(np.histogram(
        [word.i for word in haystack 
         if word.text.lower() == needle], bins=50)[0])



# I have no idea why I have to keep running this. 
# rcParams['figure.figsize'] = 16, 8

# pd.DataFrame(
#     { name: locations(name.lower(), pride) 
#      for name in ['Elizabeth', 'Darcy', 'Jane', 'Bennet']}
# ).plot(subplots=True)


## Entitites 

# print (set([w.label_ for w in grail.ents]))

# print [ent for ent in grail.ents if ent.label_ == 'WORK_OF_ART']

# print [ent for ent in grail.ents if ent.label_ == 'GPE']

# print set(list([ent.string.strip() for ent in grail.ents if ent.label_ == 'ORG']))

# print set([ent.string for ent in grail.ents if ent.label_ == 'NORP'])

# frenchPeople = [ent for ent in grail.ents if ent.label_ == 'NORP' and ent.string.strip() == 'French']
# print [ent.sent for ent in frenchPeople]


### POS tags 

tagDict = {w.pos: w.pos_ for w in pride} 
print tagDict

grailPOS = pd.Series(grail.count_by(spacy.attrs.POS))/len(grail)
pridePOS = pd.Series(pride.count_by(spacy.attrs.POS))/len(pride)

# rcParams['figure.figsize'] = 16, 8
# df = pd.DataFrame([grailPOS, pridePOS], index=['Grail', 'Pride'])
# df.columns = [tagDict[column] for column in df.columns]
# df.T.plot(kind='bar')

# prideAdjs = [w for w in pride if w.pos_ == 'PRON']
# print Counter([w.string.strip() for w in prideAdjs]).most_common(10)

# grailAdjs = [w for w in grail if w.pos_ == 'PRON']
# print Counter([w.string.strip() for w in grailAdjs]).most_common(10)

robinSents = [sent for sent in grail.sents if 'Sir Robin' in sent.string]
r2 = robinSents[2]
# for word in r2: 
#     print(word, word.tag_, word.pos_)


# print [prop for prop in dir(r2) if not prop.startswith('_')]

print r2.root
print r2.sentiment
# print list(r2.root.children)

# for word in r2: 
#     print(word, ': ', str(list(word.children)))


for sent in robinSents: 
    for word in sent: 
        if 'Robin' in word.string: 
            for child in word.children: 
                if child.pos_ == 'ADJ':
                    print(child)

# print Counter([w.string.strip() for w in pride.ents if w.label_ == 'PERSON']).most_common(10)

def adjectivesDescribingCharacters(text, character):
    sents = [sent for sent in pride.sents if character in sent.string]
    adjectives = []
    for sent in sents: 
        for word in sent: 
            if character in word.string:
                for child in word.children: 
                    if child.pos_ == 'ADJ': 
                        adjectives.append(child.string.strip())
    return Counter(adjectives).most_common(10)


# adjectivesDescribingCharacters(pride, 'Darcy')
# elizabethSentences = [sent for sent in pride.sents if 'Elizabeth' in sent.string]
# print elizabethSentences[3]



def verbsForCharacters(text, character):
    sents = [sent for sent in pride.sents if character in sent.string]
    charWords = []
    for sent in sents: 
        for word in sent: 
            if character in word.string: 
                charWords.append(word)
    charAdjectives = []
    for word in charWords: 
        # Start walking up the list of ancestors 
        # Until we get to the first verb. 
        for ancestor in word.ancestors: 
            if ancestor.pos_.startswith('V'): 
                charAdjectives.append(ancestor.lemma_.strip())
    return Counter(charAdjectives).most_common(20)

darcyVerbs = verbsForCharacters(pride, 'Darcy')
janeVerbs = verbsForCharacters(pride, 'Jane')


def verbsToMatrix(verbCounts): 
    """ 
    Takes verb counts given by verbsForCharacters 
    and makes Pandas Series out of them, suitabe for combination in 
    a DataFrame. 
    """
    return pd.Series({t[0]: t[1] for t in verbCounts})

# verbsDF = pd.DataFrame({'Elizabeth': verbsToMatrix(elizabethVerbs), 
#                         'Darcy': verbsToMatrix(darcyVerbs), 
#                         'Jane': verbsToMatrix(janeVerbs)}).fillna(0)
# verbsDF.plot(kind='bar', figsize=(14,4))




probabilities = [word.prob for word in grail] 
# pd.Series(probabilities).hist()

print list(set([word.string.strip().lower() for word in grail if word.prob < -19]))[:20]



coconut, africanSwallow, europeanSwallow, horse = nlp('coconut'), nlp('African Swallow'), nlp('European Swallow'), nlp('horse')
coconut.similarity(horse)
africanSwallow.similarity(europeanSwallow)


prideNouns = [word for word in pride if word.pos_.startswith('N')][:150]
prideNounVecs = [word.vector for word in prideNouns]
prideNounLabels = [word.string.strip() for word in prideNouns]

prideNounVecs[0].shape

lsa = TruncatedSVD(n_components=2)
lsaOut = lsa.fit_transform(prideNounVecs)

xs, ys = lsaOut[:,0], lsaOut[:,1]
for i in range(len(xs)): 
    plt.scatter(xs[i], ys[i])
    plt.annotate(prideNounLabels[i], (xs[i], ys[i]))


tfidf = TfidfVectorizer(input='filename', decode_error='ignore', use_idf=False)
