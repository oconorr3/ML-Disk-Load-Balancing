"""
Module to initialize the neural network.
"""
# Global imports:
import pickle

import config
from data import *

size_classes = config.size_classes
inputs = []
outputs = []

def skl_training(method, sklmethod):
    sklmethod.fit(inputs, outputs)
    print("Accuracy score (on given inputs and outputs): %f" % sklmethod.score(inputs, outputs))


def initialize(mode, file_list, sklmethods):

    data_size = 0

    # Retrieve information from the input files:
    for file in file_list:
        print(file)
        with open(file) as data_file:
            data = pickle.load(data_file)
            for i in range(0, len(data)):
                if i != len(data) - 1 and i > config.num_input_lines:
                    previous_rows = [data[i - x] for x in range(1, config.num_input_lines)]
                    inputs.append(parse_input_row(data[i], previous_rows))
                    outputs.append(parse_output_row(data[i + 1]))
                data_size += 1

    print("Total number of data points: %d" % data_size)

    # Normalize the inputs:
    inputs_normalized = normalize_data(inputs)

    if mode == 'train':
        print("Now training the neural network...")
        print(sklmethods)
        for method in sklmethods:
            training_network = config.sklmethods[method]

            print("Training using method \"%s\"..." % method)
            skl_training(method, training_network)
            print("Training with method \"%s\" complete!" % method)
    else:
        pass

    # if mode == 'train':
    #     print("Now training the neural network...")
    #     clf = use_sgd_classifier(max_iter=100, learning_rate='invscaling', n_jobs=-1, verbose=True,
    #             inputs=inputs_normalized, outputs=outputs)
    #     results = clf.predict(inputs_normalized)
    #     print("Writing to output.dat and measuring accuracy...")
    #     correct_count = 0
    #     with open('output.dat', 'w') as outfile:
    #         for i in range(0, len(results)):
    #             outfile.write("%s | %s | %s\n" % (inputs[i], results[i], outputs[i]))
    #             if results[i] == outputs[i]:
    #                 correct_count += 1
    #     outfile.close()
    #     print("Done!")
    #
    #     print("Accuracy: %f" % (1.0 * correct_count / len(results)))
    # else:
    #     print("Now testing the neural network...")
