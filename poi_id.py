#!/usr/bin/python

import matplotlib.pyplot 
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
import sys
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### look at data
#print len(data_dict.keys())
#print data_dict['BUY RICHARD B']
#print data_dict.values()
# print "The 21 features are listed below:"
# k=1
# for i in data_dict["SKILLING JEFFREY K"]:
#     print "Feature "+str(k)+": "+i
#     k = k+1

### remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### uncomment for printing top 4 salaries
### print outliers_final


### plot features
# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     matplotlib.pyplot.scatter( salary, bonus )

# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()

### Linear Regression to predict Bonus fromm salary
data = featureFormat(data_dict, features,remove_any_zeroes=True)
target, feature = targetFeatureSplit( data )

from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.5, random_state=42)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(feature_train,target_train)
try:
    matplotlib.pyplot.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

print reg.coef_
print reg.score(feature_test , target_test)


for feature, target in zip(feature_test, target_test):
    matplotlib.pyplot.scatter( feature, target, color="r" ) 
for feature, target in zip(feature_train, target_train):
    matplotlib.pyplot.scatter( feature, target, color="b" ) 
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def listing_dictionary(key,nor):
    a=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][nor]=="NaN":
            a.append(0.)
        elif data_dict[i][key]>=0:
            a.append(float(data_dict[i][key])/float(data_dict[i][nor]))
    return a

### creating two lists of new features
fraction_from_poi_email=listing_dictionary("from_poi_to_this_person","to_messages")
fraction_to_poi_email=listing_dictionary("from_this_person_to_poi","from_messages")

### inserting new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1


### store to my_dataset for easy export below
my_dataset = data_dict


#plot new features
for item in data_dict:
    Fraction_to=data_dict[item]['from_this_person_to_poi']
    Fraction_From=data_dict[item]['from_poi_to_this_person']
    if(data_dict[item]['poi']==1):
       matplotlib.pyplot.scatter( Fraction_From, Fraction_to,color='r' )
    else:
       matplotlib.pyplot.scatter( Fraction_From, Fraction_to,color='b' )
matplotlib.pyplot.xlabel("from_poi_to_this_person")
matplotlib.pyplot.ylabel("from_this_person_to_poi")
matplotlib.pyplot.show()

for item in data_dict:
    Fraction_to=data_dict[item]['fraction_to_poi_email']
    Fraction_From=data_dict[item]['fraction_from_poi_email']
    if(data_dict[item]['poi']==1):
       matplotlib.pyplot.scatter( Fraction_From, Fraction_to,color='r' )
    else:
       matplotlib.pyplot.scatter( Fraction_From, Fraction_to,color='b' )
matplotlib.pyplot.xlabel("fraction_from_poi_email")
matplotlib.pyplot.ylabel("fraction_to_poi_email")
matplotlib.pyplot.show()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#when Training and testing data is not split
from sklearn.tree import DecisionTreeClassifier
# clf=DecisionTreeClassifier()
# clf.fit(features,labels)
# pred=clf.predict(features)


# from sklearn.metrics import accuracy_score
# acc=accuracy_score(pred,labels)
# print acc

#after splitting of data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# accuracy = accuracy_score(pred,labels_test)
# print "Accuracy when using Naive Bayes Classifier:"+str(accuracy)
# print "Precision: " +str(precision_score(pred,labels_test))
# print "Recall: "+str(recall_score(pred,labels_test))
# print "NB algorithm time:", round(time()-t0, 3), "s"

t0 = time()

clf=DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score
acc=accuracy_score(pred,labels_test)

print "Accuracy when using Decision Tree Classifier: " + str(acc)
print "DT algorithm time:", round(time()-t0, 3), "s"
print "Precision: " +str(precision_score(pred,labels_test))
print "Recall: "+str(recall_score(pred,labels_test))

#parameter Tuning
def dt_min_samples_split(k):
    t0 = time()

    clf=DecisionTreeClassifier(min_samples_split=k)
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)

    from sklearn.metrics import accuracy_score,precision_score,recall_score
    acc=accuracy_score(pred,labels_test)

    print "Accuracy when using Decision Tree Classifier: " + str(acc)
    print "DT algorithm time:", round(time()-t0, 3), "s"
    print "Precision: " +str(precision_score(pred,labels_test))
    print "Recall: "+str(recall_score(pred,labels_test))
    
dt_min_samples_split(2)
dt_min_samples_split(3)
dt_min_samples_split(5)
dt_min_samples_split(10)
dt_min_samples_split(15)
dt_min_samples_split(20)

test_classifier(clf, my_dataset, features_list, folds = 1000)
dump_classifier_and_data(clf, my_dataset, features_list)