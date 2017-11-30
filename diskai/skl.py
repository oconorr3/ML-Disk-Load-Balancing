"""
Module to initialize the neural network.
"""
# Global imports:
import pickle
# import matplotlib.pyplot as plt

import config
from data import *

size_classes = config.size_classes
inputs = []
outputs = []

def skl_training(method, sklmethod, ins, outs):
    sklmethod.fit(ins, outs)
    print("Accuracy score (on given inputs and outputs): %f" % sklmethod.score(inputs, outputs))

# Modified from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
from sklearn.learning_curve import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    return plt

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
    normalized_inputs = normalize_data(inputs)

    if mode == 'train':
        print("Now training the neural network...")
        print(sklmethods)
        for method in sklmethods:
            training_network = config.sklmethods[method]
            print("Training using method \"%s\"..." % method)
            skl_training(method, training_network, normalized_inputs, outputs)
            # plot_learning_curve(training_network, 'Learning curve for %s' % method,
                # normalized_inputs, outputs).show()
            print("Training with method \"%s\" complete!" % method)

            print("Writing to pickle file to represent the network...")
            from sklearn.externals import joblib
            joblib.dump(training_network, 'datadump.pkl')

            print("Writing to output file...")
            with open(method + '.output', 'w') as outfile:
                outfile.write("Accuracy[%s]: %s"% (method, training_network.score(inputs, outputs)))
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
