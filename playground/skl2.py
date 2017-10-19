#!/usr/bin/env python
import json
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
import random 
import sys
import pickle

# Fill in the inputs:
inputs = []
outputs = []

def parse_input_row(row, previous_rows):
    return [0, 0, row['isWrite'], row['size'], 
            0, 0, previous_rows[0]['isWrite'], previous_rows[0]['size'],
            0, 0, previous_rows[1]['isWrite'], previous_rows[1]['size']]
    
    # return [0, 0, row['isWrite'], int(row['size'], 16)]
def parse_output_row(row):
    return [row['size']]
    # return row['size']
    
print("Eating the pickle...")
for filename in sys.argv[1:]:
    print(filename)
    with open(filename) as data_file:
        data = pickle.load(data_file)
        print(len(data))
        print("\n")
        print(data[0]['isWrite'])
        for i in range(0, len(data)):
            if i != len(data) - 1 and i > 2:
                inputs.append(parse_input_row(data[i], [data[i - 1], data[i - 2]]))
                outputs.append(parse_output_row(data[i + 1]))
print("Done tickling the pickle.")

print("Creating the neural network. Starting training...")
#clf = MLPRegressor(hidden_layer_sizes=(250,150,40),  activation='relu', solver='adam', alpha=0.001,batch_size='auto',
#               learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=100000, shuffle=True,
#               random_state=None, tol=0.0000001, verbose=True, warm_start=False, momentum=0.9,
#               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#               epsilon=1e-08)

clf = SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
       fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
       loss='squared_loss', max_iter=5, n_iter=None, penalty='l2',
       power_t=0.25, random_state=None, shuffle=True, tol=None,
       verbose=0, warm_start=False)

    
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
