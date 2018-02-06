from nltk.classify.util import *
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples
from nltk import FreqDist
import nltk
import string
import json
nltk.download('twitter_samples')


def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")] #elimina le parole "inutili" come preposizioni ecc
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

neg_tweets = []
strings_neg = twitter_samples.strings('negative_tweets.json')
strings_neg = json.dumps(strings_neg)
strings_neg = ''.join(item for item in strings_neg if not (item.startswith('"@') or item.startswith('@') or item.startswith('http'))) #rimuove tag e link
strings_neg = strings_neg.replace(":", "").replace(")", "").replace("(", "")
strings_neg = strings_neg.split()
fdist_neg =  FreqDist(strings_neg)
word_min_freqdist = list(filter(lambda x: x[1]<=5,fdist_neg.items()))
word_max_freqdist = list(filter(lambda x: x[1]>=95,fdist_neg.items()))

for word in list(strings_neg):  # rimuove le parole che hanno freq_dist >95 e <5 (natura language processing with python)
    if word in word_max_freqdist or word in word_min_freqdist :
        strings_neg.remove(word)


#in questo modo le parole prese in considerazione si riducono di circa 7 volte


for word in strings_neg[:10000]:
    neg_tweets.append((create_word_features(strings_neg), "negative"))
    print(neg_tweets)

pos_tweets = []
strings_pos = twitter_samples.strings('positive_tweets.json')
strings_pos = json.dumps(strings_pos)
strings_pos = ''.join(item for item in strings_pos if not (item.startswith('"@') or item.startswith('"@') or item.startswith('http'))) #rimuove tag e link
strings_pos = strings_pos.replace(":", "").replace(")", "").replace("(", "")
strings_pos = strings_pos.split()
fdist_pos =  FreqDist(strings_pos)
word_min_freqdist = list(filter(lambda x: x[1]<=5,fdist_pos.items()))
word_max_freqdist = list(filter(lambda x: x[1]>=95,fdist_pos.items()))


for word in list(strings_pos):  # rimuove le parole che hanno freq_dist >95 e <5 (natura language processing with python)
    if word in word_max_freqdist or word in word_min_freqdist :
        strings_pos.remove(word)


for word in strings_pos[:10000]:
    pos_tweets.append((create_word_features(strings_pos), "positive"))
    print(pos_tweets)


#implementiamo il training set e il test set in modo da avere rispettivamente 8000 e 2000 campioni
train_set = neg_tweets[:4000] + pos_tweets[:4000]
test_set =  neg_tweets[4000:] + pos_tweets[4000:]
classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print accuracy



tweet_toy = 'Coolest fan I have ever seen. I have enjoyed the game. It was an amazing night. The team has played a beautiful game against Toronto. I am proud of them. And now come to win another important championship.'
print(tweet_toy)


words = word_tokenize(tweet_toy)     #esempio di tweet positivo---->risposta algoritmo:positivo
words = create_word_features(words)
result = classifier.classify(words)
print result


'''


tweet_toy = 'The worst film I have ever seen. It was very horrible. The plot did not make sense and the shots were bad. I would not recommend this movie to anyone.'
print(tweet_toy)


words = word_tokenize(tweet_toy)
words = create_word_features(words) #esempio di tweet negativo---->risposta algoritmo:
result = classifier.classify(words)
print result 


'''