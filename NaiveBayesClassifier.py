from nltk.classify.util import *
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples
import nltk
import string
import json
nltk.download('twitter_samples')



def create_word_features(words):

    useful_words = [word for word in words if word not in stopwords.words("english")] #elimina le parole "inutili" come preposizioni ecc
    #useful_words.replace(":", "").replace(")", "").replace("(", "")   #elimina le emoticons
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

neg_tweets = []
strings_neg = twitter_samples.strings('negative_tweets.json')
strings_neg = json.dumps(strings_neg)
strings_neg = string.split(strings_neg)


for word in strings_neg[:80]:
    neg_tweets.append((create_word_features(strings_neg), "negative"))
    print(neg_tweets)

pos_tweets = []
strings_pos = twitter_samples.strings('positive_tweets.json')
strings_pos = json.dumps(strings_pos)
strings_pos = string.split(strings_pos)

for word in strings_pos[:80]:
    pos_tweets.append((create_word_features(strings_pos), "positive"))
    print(pos_tweets)




#implementiamo il training set e il test set in modo da avere rispettivamente 8000 e 2000 campioni
train_set = neg_tweets[:60] + pos_tweets[:60]
test_set =  neg_tweets[60:] + pos_tweets[60:]
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



tweet_toy = 'The worst film I have ever seen. It was very horrible. It was an amazing night. The plot did not make sense and the shots were bad. I would not recommend this movie to anyone.'
print(tweet_toy)


words = word_tokenize(tweet_toy)
words = create_word_features(words) #esempio di tweet negativo---->risposta algoritmo:
result = classifier.classify(words)
print result 
'''