from sklearn.model_selection import KFold
from sklearn import neighbors, tree, svm
from sklearn.metrics import precision_score, f1_score, roc_auc_score
import numpy as np
# import evaluation
# from evaluation import get_precision_scores

def run_knn(train_X, train_Y, options):
    n_neighbors = options['n_neighbors']
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(train_X, train_Y)
    return clf

def run_decision_tree(train_X, train_Y, options):
    depth = options['depth']
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(train_X, train_Y)
    return clf

def run_svm(train_X, train_Y, options):
    kernel = options.get('kernel', 'linear')
    C = options.get('C', 1)
    decision_function_shape = options.get('decision_function_shape', 'ovo')
    gamma = options.get('gamma', 'scale')
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma,
                  decision_function_shape=decision_function_shape)
    clf = clf.fit(train_X, train_Y)
    return clf

algos = {'knn': run_knn, 'decision_tree': run_decision_tree, 'svm': run_svm}

def kfold_cross_validation(k, train_X, train_Y, algo, options):
    print(f'Running {k}-fold cross validation for {algo} with {str(options)}')
    kf = KFold(n_splits=k)

    accuracies = []
    classifiers = []

    for train_index, test_index in kf.split(train_X):
        cf_train_X = [train_X[index] for index in train_index]
        cf_train_Y = [train_Y[index] for index in train_index]
        cf_test_X = [train_X[index] for index in test_index]
        cf_test_Y = [train_Y[index] for index in test_index]
        clf = algos[algo](cf_train_X, cf_train_Y, options)
        # evaluation = get_precision_scores(clf, cf_test_X, cf_test_Y)
        accuracy = clf.score(cf_test_X, cf_test_Y)
        accuracies.append(accuracy)
        classifiers.append(clf)
        print(f'Split accuracy: {str(accuracy)}')

    assert(len(accuracies) == k)
    # average_accuracy = sum(evaluation['accuracy'] for evaluation in evaluations) / k
    average_accuracy = sum(accuracies) / k
    print(
        f'Completed {k}-fold cross validation for {algo} with {str(options)}')
    print(f'Obtained average accuracy of: {average_accuracy}\n')
    best_classifier_idx = np.argmax(accuracies)
    return (classifiers[best_classifier_idx], max(accuracies))

def get_precision_scores(clf, test_X, test_Y):
  pred_Y = clf.predict(test_X)
  accuracy = clf.score(test_X, test_Y)
  macro_avg = precision_score(test_Y, pred_Y, average='macro', labels=np.unique(pred_Y))
  f1_score_macro = f1_score(test_Y, pred_Y, average='macro', labels=np.unique(pred_Y))
  micro_avg = precision_score(test_Y, pred_Y, average='micro', labels=np.unique(pred_Y))
  f1_score_micro = f1_score(test_Y, pred_Y, average = 'micro',labels=np.unique(pred_Y))
  rocauc_score = roc_auc_score(test_Y, clf.predict_proba(test_X), multi_class='ovr')
  return {
    'accuracy': accuracy,
    'macro_avg': macro_avg,
    'f1_score_macro': f1_score_macro,
    'micro_avg': micro_avg,
    'f1_score_micro': f1_score_micro,
    'roc_auc_score': rocauc_score
  }

def visualize_roc_auc_curve():
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    pass