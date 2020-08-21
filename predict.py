import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def malicious(url):
    logit = joblib.load('logit.model')
    vectorizer = joblib.load('vectorizer.pkl')
    result = logit.predict(vectorizer.transform([url]))
    return (result[0] == 'bad')


#simple demonstration

if __name__ == '__main__':
    print (malicious('google.com'))
    print (malicious('stock888.cn'))
