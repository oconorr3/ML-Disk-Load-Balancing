"""
Data processing methods.
"""
################################################
# Imports
################################################
# Global project imports:
import numpy as np
from sklearn.learning_curve import learning_curve

# Local project imports:
import config

################################################
# Constants
################################################
# The Scaler to use for normalization:
scaler = config.scaler

################################################
# Functions
################################################
def parse_input_row(row, previous_rows):
    """
    Parse information from a row of input. Includes previous rows up to config.num_input_lines.
    PARAMETERS:
        row : the input row to parse data from
        previous_rows : the previous rows to parse data from
    RETURNS: list of information important to this particular row of input
    """
    row_list = [row['size']]
    for i in range(config.num_input_lines - 1):
        row_list = row_list + [previous_rows[i]['size']]
    return row_list

def parse_output_row(row):
    """
    Parse information from an output row.
    PARAMETERS:
        row : the output row to parse data from
    RETURNS: size information important to this particular row of output
    """
    return config.size_classes(row['size'])

def normalize_data(inputs):
    """
    Normalize the data from inputs using the scaler given at config.scaler.
    PARAMETERS:
        inputs : list of inputs whose data to normalize
    RETURNS: inputs, normalized using the given scaler
    """
    np_inputs = np.asarray(inputs)
    #scaler.fit(np_inputs[:, [i * 4 - 1 for i in range(1, config.num_input_lines + 1)]])
    scaler.fit(np_inputs)
    inputs_normalized = scaler.transform(np_inputs)
    #    np_inputs[:, [i * 4 - 1 for i in range(1, config.num_input_lines + 1)]])
    #all_normed = np.concatenate((np_inputs[:, [0, 1, 2]], inputs_normalized[:, [0]]), axis=1)
    #all_normed = np.concatenate((all_normed, inputs_normalized[:, [0]]), axis=1)
    #for i in range(2, config.num_input_lines + 1):
    #    x = i * 4
    #    #all_normed = np.concatenate((all_normed, np_inputs[:, [x - 4, x - 3, x - 2]]), axis=1)
    #    all_normed = np.concatenate((all_normed, inputs_normalized[:, [i - 1]]), axis=1)
    return inputs_normalized

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
