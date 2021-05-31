# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import nltk
import re
nltk.download('stopwords')

data=pd.read_csv(r'/home/karthikeyan/NLP Project/SpamClassifier-master/smsspamcollection/SMSSpamCollection',
                 sep='\t',names=['labels','message'])

from nltk.corpus import stopwords
from nltk.stem.porter import  PorterStemmer
from nltk.stem import WordNetLemmatizer 
wnl = WordNetLemmatizer()
pst=PorterStemmer()

corpus=[]
#Data cleaning 
for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['message'][i])
    review=review.lower()
    review=review.split()
    
    review=[wnl.lemmatize(word) for word in review 
            if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

# creating bag of  words  Types: TfidfVectorizer & Countvectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# cv=TfidfVectorizer()
# X=cv.fit_transform(corpus).toarray()
# print(X.shape)

from sklearn.feature_extraction.text import CountVectorizer
tf=CountVectorizer()
D=tf.fit_transform(corpus).toarray()
print(D.shape)

y=pd.get_dummies(data['labels'])
y=y.iloc[:,-1]

# Train Test split

# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.model_selection import train_test_split
D_train,D_test,y_train,y_test=train_test_split(D,y,test_size=0.20,random_state=0)

# Training model using Navie Baises

from sklearn.naive_bayes import MultinomialNB
spam_dectection_model=MultinomialNB().fit(D_train,y_train)

#Predict the model

y_pred=spam_dectection_model.predict(D_test)


# Confusion Matrix
print(D_test.shape,y_pred.shape)
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)


  # accuracy_score
from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_pred)















 
    
    
    
