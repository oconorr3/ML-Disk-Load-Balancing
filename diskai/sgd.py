"""
Methods using Stochastic Gradient Descent (SGD). In this module, we use SGDRegressor to perform the
training and testing.
"""
from sklearn.linear_model import SGDClassifier

def use_sgd_classifier(max_iter, learning_rate, n_jobs, verbose, inputs, outputs):
    clf = SGDClassifier(alpha=0.0001, average=False, epsilon=0.05, eta0=0.0025, fit_intercept=False,
            l1_ratio=0.10, loss='hinge', penalty='elasticnet', power_t=0.05, random_state=None,
            shuffle=True, tol=None, warm_start=False,
            max_iter=max_iter, learning_rate=learning_rate, n_jobs=n_jobs, verbose=verbose)
    print("Using SGDClassifier. Begining the training...")
    clf.fit(inputs, outputs)
    # from sklearn.externals import joblib
    # joblib.dump(clf, 'datadump.pkl')
    print("Finished training.")

    return clf
