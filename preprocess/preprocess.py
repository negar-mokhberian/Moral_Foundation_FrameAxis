import re

import nltk

sno = nltk.stem.SnowballStemmer('english')
stop_words = [' rt ', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't"]

def isNaN(string):
    return string != string

def _sanitize(text: str) -> str:
    """Cleans text by removing whitespace, newlines and tabs
    """
    sanitized_text = " ".join(text.strip().split())
    return sanitized_text


def deEmojify(text):
    if isNaN(text): return ""
    
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(tweets):
    '''remove hashtags and urls'''
    tweets = tweets.str.replace(r'(?:\@|https?\://)\S+', ' ', regex=True)
    tweets = tweets.apply(deEmojify)
    tweets = tweets.str.lower()
    tweets = tweets.str.replace('rt :', '')
    tweets = tweets.str.replace(r'[^\w\s]', ' ', regex=True)
    tweets = tweets.str.replace(r'[^a-zA-z\s]', ' ', regex=True)  # r'[^a-zA-z0-9\s]'
    
    '''remove stopwords'''
    for s_word in stop_words:
        tweets = tweets.str.replace(' ' + s_word + ' ', ' ')
    tweets = tweets.apply(_sanitize)
    return tweets


def remove_stopwords(tweet):
    return ' '.join([w for w in tweet.split() if w not in stopwords])
