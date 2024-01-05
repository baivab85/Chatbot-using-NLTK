import nltk
import string
import random

f=open('data.txt','r',errors='ignore')
raw_doc=f.read()

raw_doc=raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
sentance = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentance_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'hey there', 'hey', 'there there')


def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
    return None


lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentance_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry. Unable to understand you"
        return robo_response
    else:
        robo_response = robo_response + sentance_tokens[idx]
        return robo_response


flag = True
print('Hello! I am the learning bot. Start typing your text')
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            flag = False
            print('Bot: You are welcome')
        else:
            greeting = greet(user_response)
            if greeting is not None:
                print('Bot: ' + greeting)
            else:
                sentance_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("Bot: ", end="")
                print(response(user_response))
                sentance_tokens.remove(user_response)
    else:
        flag = False
        print("Bot: Goodbye")
