import joblib
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")

data.head()

y = data["label"]

url_list = data["url"]

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(url_list)

joblib.dump(vectorizer,'vectorizer.pkl')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =.2, random_state = 10086)

logit = LogisticRegression()

logit.fit(x_train, y_train)

print("Accuracy: ",logit.score(x_test, y_test))

joblib.dump(logit,'logit.model')
