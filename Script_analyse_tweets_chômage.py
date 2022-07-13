# Import des librairies 

import os
import pandas as pd
import numpy as np
import tweepy as tw
from tweepy import OAuthHandler
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse
from requests_oauthlib import OAuth1
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Import des librairies pour traitement des mots les plus récurrents 
from collections import Counter
from wordcloud import WordCloud

# Import librairie permettant de réaliser des graphiques
import matplotlib.pyplot as plt

# Import librairies pour le clustering
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# Import des librairies pour la lemmatisation
import spacy.cli
spacy.cli.download("fr_core_news_md")
import spacy
from spacy import displacy
nlp = spacy.load('fr_core_news_md')
import pandas as pd

# Import librairies pour la méthode supervisée
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#Twitter API authentication
consumer_key = 'KGb77jcbxbjEHCAVj3qBXzFla'
consumer_secret = 'YEG4fokLJHSU2Dvp0erMM8vpQpJPjvl4KtA76N6YEm4VmYukfE'
access_token = '1452613788243464194-pEfk0yuGNvkEi5PdfxBCLNPnskdUch'
access_secret = 'OGefBNyYMVfAxpyzVU2XuPNRDmGvFvZ7gNgFhYKCGCx1A'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tw.API(auth, wait_on_rate_limit=True)

# Code permettant de supprimer les http (à faire appel plus tard lors du nettoyage de la base de données)
import re
pattern_http = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Code permettant de supprimer les @ (à faire appel plus tard lors du nettoyage de la base de données)
pattern_a = re.compile('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Code permettant de supprimer les # (à faire appel plus tard lors du nettoyage de la base de données)
pattern_hashtag = re.compile('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Définir la date et le mot à chercher pour l'api
search_words = "chômage"
date_since = "2021-09-01"
new_search = search_words + " -filter:retweets"
new_search

#utilisation de l'api avec les caractéristiques choisis
tweets_1 = tw.Cursor(api.search,
              q=new_search,
              lang="fr",
              since=date_since).items(1000)

users_locs = [[tweet.user.screen_name, tweet.user.location,tweet.text] for tweet in tweets_1]
users_locs

#transformation des données récupérés par datafrme
tweet_text = pd.DataFrame(data=users_locs, 
                    columns=['user', "location","tweet"])
tweet_text

# Code permettant de supprimer les http (à faire appel plus tard lors du nettoyage de la base de données)
import re
pattern_http = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Code permettant de supprimer les @ (à faire appel plus tard lors du nettoyage de la base de données)
pattern_a = re.compile('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Code permettant de supprimer les # (à faire appel plus tard lors du nettoyage de la base de données)
pattern_hashtag = re.compile('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Nettoyage de la base de données 
chomage = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweet_text['tweet'].tolist()))))
chomage = chomage.lower()
print(chomage)

# Création NLP
nlp_chom = nlp(chomage)

# Tokenisation + Lemmatisation
Lemmatisation_chom = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage' and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chom)

#compter les mots importants
all_words_no_urls = list(itertools.chain(*Lemmatisation_chom))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

#suppression des mots communs
clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

#histogramme des mots importants
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Including All Words)")

plt.show()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_EM = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_EM.generate_from_frequencies(counts_no_urls)

WordCloud_EM.to_image()

# Les données étant déjà tokeniser et lemmatiser je dis à ma fonction de ne pas refaire cela
def dummy_fun(doc):
    return doc

# Fonction qui va permettre de vectoriser les textes pour pouvoir l'analyser par le biais du k-means
vectorizer = TfidfVectorizer(tokenizer=dummy_fun,
                preprocessor=dummy_fun,
    token_pattern=None)
X = vectorizer.fit_transform(Lemmatisation_chom)

### méthode du coude

Sum_of_squared_distances = []
K = range(2,10)
for k in K:
   km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   km = km.fit(X)
   Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#k-means

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10)
print(model.fit(X))

#k-means

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")

####acp######

random_state = 0 
cls = MiniBatchKMeans(n_clusters=2, random_state=random_state)
cls.fit(X)

# reduce the features to 2D
pca = PCA(n_components=2, random_state=random_state)
reduced_features = pca.fit_transform(X.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)

####acp######

plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(X))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')



#installation du package snscrape
pip install snscrape

#import du package
import snscrape.modules.twitter as sntwitter

"""# Analyse des tweets des candidats"""

#####  MACRON #######




# création d'une liste pour mettre les données des tweets
tweets_list1 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:EmmanuelMacron) until:2022-04-01 since:2021-09-01').get_items()): 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list1.append([tweet.date, tweet.id, tweet.content, tweet.username]) 
    
# création du dataframe
tweets_EM = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


                                                                       ##### LE PEN #######
# création d'une liste pour mettre les données des tweets
tweets_list2 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:MLP_officiel) until:2022-04-01 since:2021-09-01').get_items()): 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.username]) 
    
# création du dataframe
tweets_LPN = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


                                                                       ##### PECRESSE #######


# création d'une liste pour mettre les données des tweets
tweets_list3 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:vpecresse) until:2022-04-01 since:2021-09-01').get_items()): #declare a username 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list3.append([tweet.date, tweet.id, tweet.content, tweet.username])
    
# création du dataframe
tweets_PECR = pd.DataFrame(tweets_list3, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


                                                                        ##### MELENCHON #######


# création d'une liste pour mettre les données des tweets
tweets_list4 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:JLMelenchon) until:2022-04-01 since:2021-09-01').get_items()): 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list4.append([tweet.date, tweet.id, tweet.content, tweet.username]) 

    # création du dataframe
tweets_MEL = pd.DataFrame(tweets_list4, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])



                                                                         ##### ZEMMOUR #######


# création d'une liste pour mettre les données des tweets
tweets_list5 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:ZemmourEric) until:2022-04-01 since:2021-09-01').get_items()): 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list5.append([tweet.date, tweet.id, tweet.content, tweet.username]) 
    
   # création du dataframe
tweets_ZEM = pd.DataFrame(tweets_list5, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

tweets_EM

tweets_LPN

tweets_PECR

tweets_MEL

tweets_ZEM

"""**Analyse textuelle des tweets de Macron**

```
# Ce texte est au format code
```


"""

# Nettoyage de la base de données 
chomageEM = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_EM['Text'].tolist()))))
chomageEM = chomageEM.lower()
print(chomageEM)

# Création NLP
nlp_chom = nlp(chomageEM)
# Tokenisation + Lemmatisation
Lemmatisation_chomEM = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage' and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomEM)

#liste des mots
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomEM))

#compter les mots importants
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

#enlever les mots communs
clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_EM = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_EM.generate_from_frequencies(counts_no_urls)

WordCloud_EM.to_image()

"""**Analyse textuelle des tweets de Mélenchon**

"""

#Méthode utilisée similaire à celle pour emmanuel macron

chomageMEL = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_MEL['Text'].tolist()))))
chomageMEL = chomageMEL.lower()
print(chomageMEL)

nlp_chom = nlp(chomageMEL)

Lemmatisation_chomMEL = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage' and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomMEL)

# List of all words across tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomMEL))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_MEL = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_MEL.generate_from_frequencies(counts_no_urls)

WordCloud_MEL.to_image()

"""**Analyse textuelle des tweets de Pécresse**"""

#Méthode utilisée similaire à celle pour emmanuel macron

chomagePECR = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_PECR['Text'].tolist()))))
chomageMEL = chomagePECR.lower()
print(chomagePECR)

nlp_chom = nlp(chomagePECR)

Lemmatisation_chomPECR = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage' and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomPECR)

# List of all words across tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomPECR))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_PECR = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_PECR.generate_from_frequencies(counts_no_urls)

WordCloud_PECR.to_image()

"""**Analyse textuelle des tweets de Le Pen**"""

#Méthode utilisée similaire à celle pour emmanuel macron

chomageLPN = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_LPN['Text'].tolist()))))
chomageLPN = chomageMEL.lower()
print(chomageLPN)

nlp_chom = nlp(chomageLPN)

Lemmatisation_chomLPN = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage' and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomLPN)

# List of all words across tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomLPN))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_LPN = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_LPN.generate_from_frequencies(counts_no_urls)

WordCloud_LPN.to_image()

"""**Analyse textuelle des tweets de Zemmour**"""

#Méthode utilisée similaire à celle pour emmanuel macron

chomageZEM = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_ZEM['Text'].tolist()))))
chomageZEM = chomageZEM.lower()
print(chomageMEL)

nlp_chom = nlp(chomageZEM)

Lemmatisation_chomZEM = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage' and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomZEM)

# List of all words across tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomZEM))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_ZEM = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_ZEM.generate_from_frequencies(counts_no_urls)

WordCloud_ZEM.to_image()

"""# Analyse des médias

"""

#####  europe 1 #######

# création d'une liste pour mettre les données des tweets
tweets_list1 = []
                                                                        
                          

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(' (from:europe1) until:2022-04-01 since:2021-09-01').get_items()):
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list1.append([tweet.date, tweet.id, tweet.content, tweet.username]) 
    
# création du dataframe
tweets_E1 = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])




                                                                        
                                                                        
  
                                                                        #####  médiavenir #######


# création d'une liste pour mettre les données des tweets
tweets_list2 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:mediavenir) until:2022-04-01 since:2021-09-01').get_items()): #declare a username 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.username]) #declare the attributes to be returned
    
# création du dataframe
tweets_MDV = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])




                                                                        
                                                                        
                                                                        #####  LIBERATION #######

# création d'une liste pour mettre les données des tweets
tweets_list3 = []

# utilisation de snscrape pour récupérer les données
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Chômage (from:libe) until:2022-04-01 since:2021-09-01').get_items()): #declare a username 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list3.append([tweet.date, tweet.id, tweet.content, tweet.username]) #declare the attributes to be returned
    
# création du dataframe
tweets_LIB = pd.DataFrame(tweets_list3, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

tweets_E1

tweets_MDV

tweets_LIB

###Analyse des tweets europe 1 ##########

# Nettoyage de la base de données 
chomageE1 = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_E1['Text'].tolist()))))
chomageE1= chomageE1.lower()
print(chomageE1)

# Création NLP
nlp_chom = nlp(chomageE1)
# Tokenisation + Lemmatisation
Lemmatisation_chomE1 = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage'and w.text != "c\\'est"and w.text != "n\\'est" and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomE1)

#création liste des tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomE1))

#compter les mots importants
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

#enlever les mots communs
clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_E1 = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_E1.generate_from_frequencies(counts_no_urls)

WordCloud_E1.to_image()

###Analyse des tweets médiavenir ##########

#méthode utilisée similaire à Europe 1

chomageMDV = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_MDV['Text'].tolist()))))
chomageMDV= chomageMDV.lower()
print(chomageMDV)

nlp_chom = nlp(chomageMDV)

Lemmatisation_chomMDV = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage'and w.text != "c\\'est"and w.text != "n\\'est" and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomMDV)

# List of all words across tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomMDV))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_MDV = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_MDV.generate_from_frequencies(counts_no_urls)

WordCloud_MDV.to_image()

###Analyse des tweets libe ##########

#méthode utilisée similaire à Europe 1

chomageLIB = pattern_hashtag.sub("", pattern_a.sub("", pattern_http.sub("", str(tweets_LIB['Text'].tolist()))))
chomageLIB= chomageLIB.lower()
print(chomageLIB)

nlp_chom = nlp(chomageLIB)

Lemmatisation_chomLIB = [[w.lemma_ for w in tokens if w.text != 'chomage' and w.text != 'ômage'  and w.text != '\\n\\n'  and w.text != "c\\'est"and w.text != "n\\'est" and w.text != '\n' and w.text != 'c'  and w.text != '\n' and w.text != '\\n'and w.text != '\\n-'and w.text != 'd' and w.text != ' ' and w.text != '  ' and w.text != '   ' and w.text != 'l' and w.text != 'y' and w.text != 'rt' and not w.is_stop and not w.is_punct and not w.like_num] for tokens in nlp_chom.sents]
print(Lemmatisation_chomLIB)

# List of all words across tweets
all_words_no_urls = list(itertools.chain(*Lemmatisation_chomLIB))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()

# Nuage de mots avec les 10 mots les plus récurrents 
WordCloud_LIB = WordCloud(width=800,
                        height=600,
                        min_font_size=14,
                        max_words=25,
                        background_color="white")

WordCloud_LIB.generate_from_frequencies(counts_no_urls)

WordCloud_LIB.to_image()