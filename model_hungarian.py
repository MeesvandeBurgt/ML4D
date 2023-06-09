###################################################
### Please do not edit the following code block ###
###################################################

import pkg_resources
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

required = {'scikit-learn'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if len(missing) > 0:
  print('There are missing packages: {}.'.format(missing))
  for missing_package in missing:
    print(
      'Please wait for package installation and execute the program again.')
    subprocess.check_call(
      [sys.executable, "-m", "pip", "install", missing_package])
  exit()


def pretty_print(df, message):
  print("\n================================================")
  print("%s\n" % message)
  print(df)
  print("\nColumn names below:")
  print(list(df.columns))
  print("================================================\n")


###################################################
# Feel free to edit below this point
###################################################

#######################
#  THE DATA  ##########
#######################
import pandas as pd
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv('data/processed.hungarian.csv')

# Print the first 5 rows of the dataframe
print("\n######################################################")
print("############### How the DATA LOOK ###############")
print("######################################################\n")

print(data.to_string()) #.head(200)

################################################
# Don't uncomment the following lines except if you are certain that you need them
#############################################
# Removing unknown values - if needed


# data = data[data.target!=0]
# data.loc[data["target"] == 4, "target"] = 3
# data.loc[data["target"] == 3, "target"] = 1
# data.loc[data["target"] == 2, "target"] = 1
# targets = ['0', '1', '2', '3', '4']
targets = ['0','1']
##############################################

print("\n######################################################")
print("############### Distribution  of our classes ###############")
print("######################################################\n")
##### Understand the distribution of the Labels ###########

#Figure is saved in the img folder


print("##############################\n\n")

#######################
# SPLITTING THE DATA  #
#######################
# Import train_test_split function
from sklearn.model_selection import train_test_split

print(
  "#########################################################################")
print("############### Splitting the Data ###############")
print(
  "#########################################################################\n\n"
)
# Split dataset into training set and test set
# X_train: features to train our model
# y_train: Labels of the training data
# X_test: features ot test our model
# y_test: labels of the testing data

import random

# Here we change remove labels that are not usable ('slope', 'ca', 'thal'), by replacing the old datafram with a new one.
used_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'target']
data = data[used_columns]

# Save 
for column in used_columns:
  data.hist(column=column)
  plt.savefig(f'img/{column}hist.pdf')

# For all features (columns) that we use for the model, we fill all NaN values with random values between the min. and max. values of the feature's distribution.
for i in used_columns:
  data[i] = data[i].fillna(random.randrange(min(data[i]),max(data[i]),1))

# Count the number of instances (datapoints) per label
instances_0 = (data['target'] == 0).sum()
print(instances_0)
instances_1 = (data['target'] == 1).sum()
print(instances_1)

# For loop that removes random samples from the label with a more frequent occurrence, until the frequency is equal to the other label.
for datapoint in range(0,instances_0):
  if instances_0 <= instances_1:
    break
  else:
    sample0 = data[data['target'] == 0].sample()
    data = data.drop(sample0.index)
    instances_0 -= 1
    # print(instances_0)

# Here we show the new dataset again to validate all empty cells are succesfully replaced with random values.
print(data.to_string()) #.head(200)


print("Check the Figure in the img folder ")

# Split features from labels
# Features
X = data[[ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']] # , 'slope', 'ca', 'thal'
# Labels
y = data['target']  # Labels

print("\n ### FEATURES ###")
print(list(X))
print("\n #### Labels ###")
l = list(y.unique())
l.sort()
print(l)
print("\n")

#  Split arrays or matrices into random train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7, test_size=0.3)  # 70% training and 30% test

# check the size of the data - How much do you use for training and how much for testing ?

print("Training Features shape (X_train): " + str(X_train.shape))
print("Training Labels shape (y_train): " + str(y_train.shape))
print("Testing Features shape (X_test): " + str(X_test.shape))
print("Testing Labels shape (y_test): " + str(y_test.shape))

print("\n ####  Save Histogram of Train/Test Label Counts ####")
plt.clf() # clear plot
y_train.hist()
y_test.hist()
plt.savefig('img/y_train_test_count.pdf')
print("Histogram Saved")

print("\n ####  Training Dataset -- Label Counts ####")
print(y_train.value_counts())

print("\n ####  Test Dataset -- Label Counts ####")
print(y_test.value_counts())

print(
  "\n#########################################################################"
)
print("############### TRAINING THE MODEL(s) ###############")
print(
  "#########################################################################\n\n"
)

# #########################
# # TRAINING THE MODEL(S) #
# #########################
# importing ML models.
#You can find more models in sklearn library
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
# #Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Dummy Classifier
# clf = DummyClassifier(strategy="constant", constant=0)

# Kneighbhors Classifier
# clf = KNeighborsClassifier(4)

#Random Forest Classifier
clf = RandomForestClassifier(n_estimators=6)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

# Prediction happens here!
y_pred = clf.predict(X_test)
print(clf)
print("\n\n")

##############################
# EVALUATION OF THE MODEL(S) #
##############################

################## CONFUSION MATRIX ######## ############# #########################
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# # SHOW
print(
  "#########################################################################")
print("############### CONFUSION MATRIX  ###############")
print(
  "#########################################################################\n\n"
)

plt.clf() # clear plot
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', ax=ax)
plt.savefig('img/confusion_matrix.pdf')
print("Saved the confusion matrix under the img folder\n")

###########################################################################

print(
  "#########################################################################")
print("############### EVALUATION METRICS  ###############")
print(
  "#########################################################################\n\n"
)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=targets))

# ################## Feature Importance ############################################
from util import computeFeatureImportance

print(
  "#########################################################################")
print("############### FEATURE IMPORTANCE  ###############")
print(
  "#########################################################################\n\n"
)

feature_importance = computeFeatureImportance(X, y, scoring="f1_weighted", model=clf)
pretty_print(feature_importance, "Display feature importance based on f1-score")

# ################## CROSS VALIDATION ############################################
from sklearn.model_selection import cross_validate
from util import printScores

print(
  "#########################################################################")
print("############### CROSS VALIDATION ###############")
print(
  "#########################################################################\n\n"
)

scores = cross_validate(clf, X, y, cv=4, scoring=('accuracy', 'precision_micro', 'recall_macro', 'f1_micro', 'f1_macro'), return_train_score=True)

printScores(scores)

# print(scores)
