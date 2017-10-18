import json
from sklearn.neural_network import MLPClassifier
import random 
import sys

# Fill in the inputs:
inputs = []
outputs = []

def parse_input_row(row):
    return [row['ioType'], 0, row['isWrite'], int(row['size'], 16)]
def parse_output_row(row):
    return [row['ioType'], row['isWrite']]

print("Opening the JSON file...")
with open(sys.argv[1]) as data_file:
    data = json.load(data_file)
    print(len(data))
    for i in range(0, len(data)):
        if i != len(data) - 1:
            inputs.append(parse_input_row(data[i]))
            outputs.append(parse_output_row(data[i + 1]))
print("Done processing JSON.")

# print("Creating the neural network. Starting training...")
# clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, max_iter=100000, learning_rate='adaptive',
#     hidden_layer_sizes=(50, 20), random_state=1, verbose=True, tol=1e-6)
# clf.fit(inputs, outputs)
# from sklearn.externals import joblib
# joblib.dump(clf, 'datadump.pkl')
# print("Done training.")

from sklearn.externals import joblib
clf = joblib.load('datadump.pkl')
results = clf.predict(inputs)
# Mapping from result data to strings:
ioType = {'0': 'FileIo', '1':'Disk'}
writeRead = {'0': 'Read', '1': 'Write'}

correct_count = 0
for i in range(0, len(results)):
#    print(results[i].tolist(), outputs[i])
    if results[i].tolist() == outputs[i]:
        correct_count += 1

print("Accuracy: %f" % (1.0 * correct_count / len(results)))
