import numpy as np
import pandas as pd

# pandas read admissions.csv
admissions = pd.read_csv("admissions.csv")

# parse rank column into dummy variables
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# standarize GRE and GPA as number of standard deviations from mean
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

# split off random 90% of the data for training
# np.random.seed(42)
sample_id = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
training_data, test_data = data.ix[sample_id], data.drop(sample_id)

# split into features and targets
training_features, training_targets = training_data.drop('admit', axis=1), training_data['admit']
test_features, test_targets = test_data.drop('admit', axis=1), test_data['admit']

# print
print("admissions: (len = {0})".format(len(admissions)))
print(admissions[0:5])
print("training_features: (len = {0})".format(len(training_features)))
print(training_features[0:5])
print("test_features: (len = {0})".format(len(test_features)))
print(test_features[0:5])
