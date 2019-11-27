# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:20:37 2019

@author: 10649929
"""
import pandas as pd

messages= pd.read_csv('C:\\Users\\10649929\\Desktop\\SMSSpamCollection',sep='\t',names=["label","message"])

#Data Cleaning and preprocessing

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]','',messages['message'][i])
    review=review.lower()
    review= review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=''.join(review)
    corpus.append(review)
    
#creating the bag of model

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1]

#Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)

#Naives Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred=spam_detect_model.predict(X_test)

#confusion matrix to know correct prediction

from sklearn.metrics import confusion_matrix
confusion_m =confusion_matrix(y_test,y_pred)

#accuracy score

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)