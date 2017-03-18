# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:45:30 2017

@author: A1144
"""
#week-3: ML Specialization
import graphlab
products=graphlab.SFrmae('amazon_baby.gl/')
products.head()

#creating the word count vector
products['word_count']=graphlab.text_analytics.count_words(products['review'])

# Exploring the most popular product:
graphlab.canvas.set_target('ipynb')
products['name'].show()

#Explore the most popular product Vulli Sophie:
giraffe_reviews=products[products['name']=='Vulli Sophie the Giraffe Teether']
len(giraffe_reviews)
giraffe_reviews['rating'].show(view='Categorical')


#Defining which review has +ve or -ve sentiment
#Build a sentiment classifier
products['rating'].show(view='Categorical')

#Define what is +ve or -ve sentment

#ignore all 3* reviews
products= products[products['rating']!=3]
#positive sentiment = 4* and 5* ratings
products['sentiment']= products['rating']>=4



#Training a sentiment classifier
train_data, test_data= products.random_split(.8, seed=0)
sentiment_model=graphlab.logistic_classifier.create(train_data, target='sentiment', features=['word_count'], validation_set=test_data)
sentiment_model.evaluate(test_data, metric='roc_curve')
sentiment_model.show(view='Evaluation') # trade off FPR AND TPR 


#Applying model to find most +ve and -ve reviews
giraffe_reviews['predicted_sentiement']=sentiment_model.predict(giraffe_reviews,output_type='probability')
#sort the reviews based on the predicted sentiment and explore
giraffe_reviews=giraffe_reviews.sort('predicted_sentiment', ascending=False)
giraffe_reviews[0]['reviews']
giraffe_reviews[1]['reviews']

#show most negative reviews
giraffe_reviews[-1]['reviews']
giraffe_reviews[-2]['reviews']

