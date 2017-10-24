#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import *
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDRegressor
import random
import sys
import pickle

# Fill in the inputs:
inputs = []
outputs = []

# Create input normalizer:
#scaler = MinMaxScaler()
scaler = QuantileTransformer(output_distribution='normal')

def parse_input_row(row, previous_rows):
    return [0, 0, row['isWrite'], row['size'],
            0, 0, previous_rows[0]['isWrite'], previous_rows[0]['size'],
            0, 0, previous_rows[1]['isWrite'], previous_rows[1]['size'],
            0, 0, previous_rows[2]['isWrite'], previous_rows[2]['size'],
            0, 0, previous_rows[3]['isWrite'], previous_rows[3]['size']]

    # return [0, 0, row['isWrite'], int(row['size'], 16)]
def parse_output_row(row):
    """
    2048   >= size => 0      -> 'small'
    4096   >= size  > 2048   -> 'medium'
    16384  >= size  > 4096   -> 'large'
    32768  >= size  > 16384  -> 'xlarge'
    131072 >= size  > 32768  -> 'huge'
    infty  >= size  > 131072 -> 'enormous'
    """
    size = row['size']
    if (size >= 0 and size <= 2048):
        return 'small'
    elif (size > 2048 and size <= 4096):
        return 'medium'
    elif (size > 4096 and size <= 16384):
        return 'large'
    elif (size > 16384 and size <= 32768):
        return 'xlarge'
    elif (size > 32768 and size <= 131072):
        return 'huge'
    else:
        return 'enormous'

print("Eating the pickle...")
for filename in sys.argv[1:]:
    print(filename)
    with open(filename) as data_file:
        data = pickle.load(data_file)
        print(len(data))
        print("\n")
        print(data[0]['isWrite'])
        for i in range(0, len(data)):
            if i != len(data) - 1 and i > 4:
                inputs.append(parse_input_row(data[i], [data[i - 1], data[i - 2], data[i - 3], data[i - 4]]))
                outputs.append(parse_output_row(data[i + 1]))
print("Done tickling the pickle.")

np_inputs = np.asarray(inputs)
np_outputs = np.asarray(outputs)
scaler.fit(np_inputs[:, [3, 7, 11, 15, 19]])
inputs_normalized = scaler.transform(np_inputs[:, [3, 7, 11, 15, 19]]) #, axis=0)
# outputs_normalized = scaler.transform(np_outputs) #, axis=0)
old_inputs = inputs
old_outputs = outputs
inputs = np.concatenate((np_inputs[:, [0, 1, 2]], inputs_normalized[:, [0]]), axis=1)
inputs = np.concatenate((inputs, np_inputs[:, [4, 5, 6]]), axis=1)
inputs = np.concatenate((inputs, inputs_normalized[:, [1]]), axis=1)
inputs = np.concatenate((inputs, np_inputs[:, [8, 9, 10]]), axis=1)
inputs = np.concatenate((inputs, inputs_normalized[:, [2]]), axis=1)
inputs = np.concatenate((inputs, np_inputs[:, [12, 13, 14]]), axis=1)
inputs = np.concatenate((inputs, inputs_normalized[:, [3]]), axis=1)
inputs = np.concatenate((inputs, np_inputs[:, [16, 17, 18]]), axis=1)
inputs = np.concatenate((inputs, inputs_normalized[:, [4]]), axis=1)

print("Creating the neural network. Starting training...")
clf = MLPClassifier(hidden_layer_sizes=(4500),  activation='relu', solver='adam', alpha=0.001, batch_size=2000,
               learning_rate='adaptive', learning_rate_init=0.75, power_t=0.5, max_iter=2000, shuffle=True,
               random_state=None, tol=0.000001, verbose=True, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

clf.fit(inputs, outputs)
from sklearn.externals import joblib
joblib.dump(clf, 'datadump.pkl')
print("Done training.")

# from sklearn.externals import joblib
# clf = joblib.load('datadump.pkl')
results = clf.predict(inputs)
# results_transformed = scaler.inverse_transform(results.reshape(-1, 1))
# with open('results.dat', 'w') as out:
#    out.write("%s" % results_transformed)
# out.close()
#print(scaler.inverse_transform([results]), outputs[0])
# Mapping from result data to strings:
ioType = {'0': 'FileIo', '1':'Disk'}
writeRead = {'0': 'Read', '1': 'Write'}

print("Writing to output.dat and measuring accuracy...")
correct_count = 0
with open('output.dat', 'w') as outfile:
    for i in range(0, len(results)):
        outfile.write("%s | %s | %s\n" % (old_inputs[i], results[i], outputs[i]))
        if results[i] == outputs[i]:
            correct_count += 1
outfile.close()
print("Done!")

print("Accuracy: %f" % (1.0 * correct_count / len(results)))
