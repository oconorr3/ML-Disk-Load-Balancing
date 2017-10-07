from sklearn.neural_network import MLPClassifier
import random 

# Fill in the inputs:
inputs = []
for i in range(0, 400):
    if (i < 100):
        inputs.append([0, 1, 0, random.randint(0,100)])
    if (i >= 100 and i < 200):
        inputs.append([0, 1, 1, random.randint(0,200)])
    if (i >=200 and i < 300):
        inputs.append([1, 1, 0, random.randint(0, 300)])
    if (i >= 300 and i < 400):
        inputs.append([1, 1, 1, random.randint(0, 400)])
outputs = []
for i in range(0, 400):
    outputs.append([inputs[(i + 1) % 399][0], inputs[(i + 1) % 399][2]])

clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, max_iter=100000, learning_rate='adaptive',
    hidden_layer_sizes=(50, 20), random_state=1, verbose=True, tol=1e-6)
clf.fit(inputs, outputs)

test_inputs = [
    [0, 1, 0, 28],
    [1, 1, 0, 250]
]
results = clf.predict(test_inputs)
# Mapping from result data to strings:
ioType = {'0': 'FileIo', '1':'Disk'}
writeRead = {'0': 'Read', '1': 'Write'}

count = 0
for result in results:
    print("Input: %s -> Output: %s" % (test_inputs[count], (ioType[str(result[0])], writeRead[str(result[1])])))
    count += 1

