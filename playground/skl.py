#!/usr/bin/env python
import json
from sklearn.neural_network import MLPRegressor
import random 
import sys
import pickle

# Fill in the inputs:
inputs = []
outputs = []

def parse_input_row(row):
    return [0, 0, row['isWrite'], row['size']]
    # return [0, 0, row['isWrite'], int(row['size'], 16)]
def parse_output_row(row):
    print(row, [row['isWrite'], row['size']])
    return [row['size']]
    # return row['size']
    
print("Eating the pickle...")
with open(sys.argv[1]) as data_file:
    data = pickle.load(data_file)
    print(len(data))
    print("\n")
    print(data[0]['isWrite'])
    for i in range(0, len(data)):
        if i != len(data) - 1:
            inputs.append(parse_input_row(data[i]))
            outputs.append(parse_output_row(data[i + 1]))
print("Done tickling the pickle.")

print("Creating the neural network. Starting training...")
clf = MLPRegressor(solver='lbfgs', activation='logistic', alpha=1e-5, max_iter=100000, learning_rate='adaptive',
    hidden_layer_sizes=(20), random_state=1, verbose=True, tol=1e-6)
clf.fit(inputs, outputs)
from sklearn.externals import joblib
joblib.dump(clf, 'datadump.pkl')
print("Done training.")

# from sklearn.externals import joblib
# clf = joblib.load('datadump.pkl')
results = clf.predict(inputs)
print([results[0]], outputs[0])
# Mapping from result data to strings:
ioType = {'0': 'FileIo', '1':'Disk'}
writeRead = {'0': 'Read', '1': 'Write'}

correct_count = 0
for i in range(0, len(results)):
    print(inputs[i], results[i])
    if [results[i]] == outputs[i]:
        correct_count += 1

print("Accuracy: %f" % (1.0 * correct_count / len(results)))
