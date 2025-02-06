#-------------------------------------------------------------------------
# AUTHOR: Brad Kim
# FILENAME: ML-Assignment-1
# SPECIFICATION: Read the file  contact_lens.csv and output a decision tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: 38 Minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)
         #print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here
X = []

# functions to convert feature into numbers
def convertAge(x):
  if x == 'Young':
    return 1
  if x == 'Prepresbyopic':
    return 2
  if x == 'Presbyopic':
    return 3
  else:
    return 1

def convertSpec(x):
  if x == 'Myope':
    return 1
  if x == 'Hypermetrope':
    return 2
  else:
    return 1

def convertAst(x):
  if x == 'Yes':
    return 1
  if x == 'No':
    return 2
  else:
    return 1

def convertTpr(x):
  if x == 'Normal':
    return 1
  if x == 'Reduced':
    return 2
  else:
    return 1

def convertRec(x):
  if x == 'Yes':
    return 1
  if x == 'No':
    return 2
  else:
    return 1

# index to feature conversion mapping
convertFunc = {
  0: convertAge,
  1: convertSpec,
  2: convertAst,
  3: convertTpr,
  4: convertRec,
}

#For each row in the database,  convert each feature to numerical value and load
for row in db:
  updated_feat_row = []
  
  for i,feat in enumerate(row[:-1]):
    changed_val = convertFunc[i](feat)
    updated_feat_row.append(changed_val)
  
  X.append(updated_feat_row)

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> add your Python code here
Y = []
for row in db:
  updated_class = []
  
  changed_val = convertFunc[len(row)-1](row[-1])
  updated_class.append(changed_val)

  Y.append(updated_class)
  
#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()