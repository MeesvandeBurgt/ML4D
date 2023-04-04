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
data = pd.read_csv('data/processed.switzerland.csv')
clev_data = pd.read_csv('data/processed.cleveland.csv')

clev_data = clev_data[['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'target']].dropna()

# Print the first 5 rows of the dataframe
print("\n######################################################")
print("############### How the DATA LOOK ###############")
print("######################################################\n")

print(data.to_string()) #.head(200)

################################################
# Don't uncomment the following lines except if you are certain that you need them
#############################################
# Removing unknown values - if needed

clev_data.loc[clev_data["target"] == 1, "target"] = 0
clev_data.loc[clev_data["target"] == 4, "target"] = 1
clev_data.loc[clev_data["target"] == 3, "target"] = 1
clev_data.loc[clev_data["target"] == 2, "target"] = 1

#data = data[data.target!=0]
data.loc[data["target"] == 1, "target"] = 0
data.loc[data["target"] == 4, "target"] = 1
data.loc[data["target"] == 3, "target"] = 1
data.loc[data["target"] == 2, "target"] = 1
# targets = ['0', '1', '2', '3', '4']
#targets = ['1', '2', '3']
targets = ['0','1']
##############################################

print("\n######################################################")
print("############### Distribution  of our classes ###############")
print("######################################################\n")
##### Understand the distribution of the Labels ###########

#Figure is saved in the img folder

data.hist(column='target')
plt.savefig(f'img/hist.pdf')
print("##############################\n\n")

del data['ca']
mask = data.isna().sum(axis=1) >= 3
data = data[~mask].sort_values('target')

print(data.to_string())
print(data.shape[0])

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

used_columns = ['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
data = data[used_columns]

for column in used_columns:
  data.hist(column=column)
  plt.savefig(f'img/{column}hist.pdf')

# #Here, the number of instances of the targets is counted.
# instances_1 = (data['target'] == 1).sum()
# print(instances_1)
# instances_2 = (data['target'] == 2).sum()
# print(instances_2)
# instances_3 = (data['target'] == 3).sum()
# print(instances_3)

# # Here, the number of datapoint of target 1 is reduced to the same number of target 2 (the least frequent target).
# for datapoint in range(0,len(data)):
#   if instances_1 <= instances_2:
#     break
#   else:
#     sample1 = data[data['target'] == 1].sample()
#     data = data.drop(sample1.index)
#     instances_1 -= 1

# # Here, the number of datapoint of target 3 is reduced to the same number of target 2 (the least frequent target).
# for datapoint in range(0,len(data)):
#   if instances_3 <= instances_2:
#     break
#   else:
#     sample3 = data[data['target'] == 3].sample()
#     data = data.drop(sample3.index)
#     instances_3 -= 1

# Count the number of instances (datapoints) per label
instances_0 = (data['target'] == 0).sum()
print('instances of 0:', instances_0)
instances_1 = (data['target'] == 1).sum()
print('instances of 1:', instances_1)

# filter for only target = 1 datapoints
target1data = data[data['target'] == 1]

# Sort the filtered DataFrame by the number of NaN values in each row
target1NaNsorted = target1data.isna().sum(axis=1).sort_values(ascending=False)
#print(target1NaNsorted.to_string())


# For loop that removes random samples from the label with a more frequent occurrence, until the frequency is equal to the other label.
for datapoint in range(0,instances_1):
  if instances_1 <= instances_0:
    break
  else:
    if len(target1NaNsorted) != 0:
      # Get the index label of the first row (i.e., the row with the largest number of NaN values)   
      idx = target1NaNsorted.idxmax()
      data = data.drop(index=idx)
      target1NaNsorted = target1NaNsorted.drop(idx)
      instances_1 -= 1
      #print(instances_0)
    else:
      sample1 = data[data['target'] == 1].sample()
      data = data.drop(sample1.index)
      instances_1 -= 1
      #print(instances_0)

print(data.to_string()) #.head(200)

# Here, random numbers are filled in the empty cells (some features have float values and require extra care).
for i in used_columns:
    if i == 'target':
        data[i] = data[i]
    elif i == 'slope':
        data[i] = data[i].fillna(random.randrange(1,4,1))
    elif i == 'oldpeak':
        min = min(data[i])
        max = max(data[i])
        print(i, min, max)
        data[i] = data[i].fillna(round(random.uniform(min,max),1))
    else:    
        try:
            data[i] = data[i].fillna(random.randrange(min(data[i]),max(data[i]),1))
            print(i, min(data[i]), max(data[i]))
        except Exception as e:
            print(i, e)    

used_features = ['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']
def normalize_age(data):
    return (data - data.min()) / (data.max() - data.min())
  
for i in used_features:
  data[i] = normalize_age(data[i])

print(data.to_string()) #.head(200)

# Split features from labels
# Features
X = data[[ 'age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']] #  'fbs', 'ca', 'chol', 'thal'
# Labels
y = data['target']  # Labels

X_test1 = clev_data[['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']]
y_test1 = clev_data['target']

print(clev_data.to_string())

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
# clf = KNeighborsClassifier(7)

#Random Forest Classifier
clf = RandomForestClassifier(n_estimators=6)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

# Prediction happens here!
y_pred = clf.predict(X_test)
# y_pred1 = clf.predict(X_test1)
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
