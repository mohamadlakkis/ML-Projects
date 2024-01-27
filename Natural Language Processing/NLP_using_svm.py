import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
importing the daataset
'''
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


'''
cleaning the texts
'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []# all the reviews
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])#^ means not
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # lec 238
  review = ' '.join(review)# adding space between each way
  corpus.append(review)
print(corpus)
'''
creating the bag of word model (tokenisation)
'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)# we removed 66 from 1566 to remove the unecssary names
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
print(len(x[0])) # this cell is to see what is the len of words  before 37 (1566)
'''
splitting the dataset into traning and test set 
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
'''
feature scalling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
'''
training the kernel SVm model on the training set
'''
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)
'''
predicting the test set result
'''
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
'''
making the confusion matrix # it will show us how many right predictions and false
'''
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
matrix = multilabel_confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test,y_pred) # pourcentage of right answers
print(matrix)
print(acc)
'''
predicting a new result # we repeat the same process again
'''
new_review = ' i hate it so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
