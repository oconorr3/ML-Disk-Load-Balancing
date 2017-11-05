"""
Machine learning configuration file.
"""
from sklearn.preprocessing import *
from sklearn.neural_network import *
from sklearn.linear_model import *

# Define size classes for disk I/O sizes:
def size_classes(size):
    """
    Definitions of size classes:
    2048   >= size => 0      -> 'small'
    4096   >= size  > 2048   -> 'medium'
    16384  >= size  > 4096   -> 'large'
    131072 >= size  > 16384  -> 'huge'
    infty  >= size  > 131072 -> 'enormous'
    """
    if (size >= 0 and size <= 2048):
        return 'small'
    elif (size > 2048 and size <= 4096):
        return 'medium'
    elif (size > 4096 and  size <= 32768):
        return 'large'
    elif (size > 32768 and size <= 131072):
        return 'huge'
    else:
        return 'enormous'

# The number of input lines to use for the
num_input_lines = 5

# The scaler to use for normalization.
scaler = QuantileTransformer(output_distribution='normal')

###########################################################
# Scikit-learn configuration for various training methods #
###########################################################
sklmethods = {
    ######################################################
    # Set up MLPClassifier
    ######################################################
    # MLPClassifier with relu activation function:
    'mlp-relu1': MLPClassifier(hidden_layer_sizes=(1500), alpha=0.001, batch_size=2000,
            learning_rate='adaptive', power_t=0.5, max_iter=2000, shuffle=True, random_state=None,
            tol=0.000001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            activation='relu', solver='adam', learning_rate_init=0.75, verbose=True),
    'mlp-relu2': MLPClassifier(hidden_layer_sizes=(1500), alpha=0.001, batch_size=2000,
            learning_rate='adaptive', power_t=0.5, max_iter=2000, shuffle=True, random_state=None,
            tol=0.000001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            activation='relu', solver='sgd', learning_rate_init=0.75, verbose=True),
    'mlp-relu2': MLPClassifier(hidden_layer_sizes=(1500), alpha=0.001, batch_size=2000,
            learning_rate='adaptive', power_t=0.5, max_iter=2000, shuffle=True, random_state=None,
            tol=0.000001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            activation='relu', solver='lbfgs', learning_rate_init=0.75, verbose=True),
    # MLPClassifier with logistic activation function:
    'mlp-logistic1': MLPClassifier(hidden_layer_sizes=(1500), alpha=0.001, batch_size=2000,
            learning_rate='adaptive', power_t=0.5, max_iter=2000, shuffle=True, random_state=None,
            tol=0.000001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            activation='logistic', solver='adam', learning_rate_init=0.75, verbose=True),
    'mlp-logistic2': MLPClassifier(hidden_layer_sizes=(1500), alpha=0.001, batch_size=2000,
            learning_rate='adaptive', power_t=0.5, max_iter=2000, shuffle=True, random_state=None,
            tol=0.000001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            activation='logistic', solver='sgd', learning_rate_init=0.75, verbose=True),
    'mlp-logistic3': MLPClassifier(hidden_layer_sizes=(1500), alpha=0.001, batch_size=2000,
            learning_rate='adaptive', power_t=0.5, max_iter=2000, shuffle=True, random_state=None,
            tol=0.000001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.90, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            activation='logistic', solver='lbfgs', learning_rate_init=0.75, verbose=True),
    ######################################################
    # Set up SGDClassifier
    ######################################################
    # SGDClassifier with
    'sgd-hinge': SGDClassifier(alpha=0.0001, average=False, epsilon=0.05, eta0=0.0025,
            fit_intercept=False, l1_ratio=0.10, learning_rate='invscaling', power_t=0.05,
            random_state=None, shuffle=True, tol=None, warm_start=False,
            verbose=True, loss='hinge', penalty='elasticnet', max_iter=1000, n_jobs=-1),
    'sgd-log': SGDClassifier(alpha=0.0001, average=False, epsilon=0.05, eta0=0.0025,
            fit_intercept=False, l1_ratio=0.10, learning_rate='invscaling', power_t=0.05,
            random_state=None, shuffle=True, tol=None, warm_start=False,
            verbose=True, loss='log', penalty='elasticnet', max_iter=1000, n_jobs=-1),
    'sgd-huber': SGDClassifier(alpha=0.0001, average=False, epsilon=0.05, eta0=0.0025,
            fit_intercept=False, l1_ratio=0.10, learning_rate='invscaling', power_t=0.05,
            random_state=None, shuffle=True, tol=None, warm_start=False,
            verbose=True, loss='huber', penalty='elasticnet', max_iter=1000, n_jobs=-1),
    'sgd-modhuber': SGDClassifier(alpha=0.0001, average=False, epsilon=0.05, eta0=0.0025,
            fit_intercept=False, l1_ratio=0.10, learning_rate='invscaling', power_t=0.05,
            random_state=None, shuffle=True, tol=None, warm_start=False,
            verbose=True, loss='modified_huber', penalty='elasticnet', max_iter=1000, n_jobs=-1),
    'sgd-perceptron': SGDClassifier(alpha=0.0001, average=False, epsilon=0.05, eta0=0.0025,
            fit_intercept=False, l1_ratio=0.10, learning_rate='invscaling', power_t=0.05,
            random_state=None, shuffle=True, tol=None, warm_start=False,
            verbose=True, loss='perceptron', penalty='elasticnet', max_iter=1000, n_jobs=-1),

}


###########################################################
# Tensorflow configuration for various training methods   #
###########################################################
