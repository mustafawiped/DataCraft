import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('veri_seti.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['metin'])

y = data['etiket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Doğruluk Oranı:", accuracy)
