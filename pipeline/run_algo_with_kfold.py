from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, tree, svm
from sklearn.model_selection import KFold
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import pickle


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
    C = options.get('C', 0.01)
    decision_function_shape = options.get('decision_function_shape', 'ovo')
    gamma = options.get('gamma', 'scale')
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma,
                  decision_function_shape=decision_function_shape, probability=True)
    clf = clf.fit(train_X, train_Y)
    return clf


def run_logistic_reg(train_X, train_Y, options):
    max_iter = options['max_iter']
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(train_X, train_Y)
    return clf


algos = {'knn': run_knn, 'decision_tree': run_decision_tree,
         'svm': run_svm, 'logistic': run_logistic_reg}


def get_model_file_name(algo, options, append_text=''):
    initial_name = f'{algo}_{str(options)}'
    processed_name = re.sub(r'[^\w]', '', initial_name).replace(' ', '_')
    file_name = f'models/{processed_name}'
    if append_text:
        file_name += f'_{append_text}'
    file_name += '.pickle'
    return file_name


def load_or_train_kfold_model(algo, options, train_X, train_Y, train_index, validation_index, append_text=''):
    model_file_name = get_model_file_name(algo, options, append_text)
    validation_X = [train_X[index] for index in validation_index]
    validation_Y = [train_Y[index] for index in validation_index]
    try:
        model = pickle.load(open(model_file_name, 'rb'))
        print(f'Loaded kfold model from file: {model_file_name}')
    except FileNotFoundError:
        cf_train_X = [train_X[index] for index in train_index]
        cf_train_Y = [train_Y[index] for index in train_index]
        model = algos[algo](cf_train_X, cf_train_Y, options)
        pickle.dump(model, open(model_file_name, 'wb'))
        print(f'Trained kfold model and saved to file: {model_file_name}')
    finally:
        # evaluation = get_precision_scores(model, cf_test_X, cf_test_Y)
        accuracy = model.score(validation_X, validation_Y)
        print(f'Split accuracy: {str(accuracy)}')
    return (model, accuracy)


def load_or_train_model(algo, options, train_X, train_Y):
    model_file_name = get_model_file_name(algo, options)
    try:
        model = pickle.load(open(model_file_name, 'rb'))
        print(f'Loaded model from file: {model_file_name}')
    except FileNotFoundError:
        model = algos[algo](train_X, train_Y, options)

        pickle.dump(model, open(model_file_name, 'wb'))
        print(f'Trained model and saved to file: {model_file_name}')
    return model


def kfold_cross_validation(k, train_X, train_Y, algo, options):
    print(f'Running {k}-fold cross validation for {algo} with {str(options)}')
    kf = KFold(n_splits=k)

    clfs_and_accuracies = [load_or_train_kfold_model(algo, options, train_X, train_Y, train_index, test_index, str(
        index)) for index, (train_index, test_index) in enumerate(kf.split(train_X))]

    accuracies = [accuracy for (model, accuracy) in clfs_and_accuracies]
    assert(len(accuracies) == k)
    # average_accuracy = sum(evaluation['accuracy'] for evaluation in evaluations) / k
    average_accuracy = sum(accuracies) / k
    print(
        f'Completed {k}-fold cross validation for {algo} with {str(options)}')
    print(f'Obtained average accuracy of: {average_accuracy}\n')
    model = load_or_train_model(algo, options, train_X, train_Y)
    return (model, max(accuracies))


def get_precision_scores(clf, test_X, test_Y):
    pred_Y = clf.predict(test_X)
    accuracy = clf.score(test_X, test_Y)
    macro_avg = precision_score(
        test_Y, pred_Y, average='macro', labels=np.unique(pred_Y))
    f1_score_macro = f1_score(
        test_Y, pred_Y, average='macro', labels=np.unique(pred_Y))
    micro_avg = precision_score(
        test_Y, pred_Y, average='micro', labels=np.unique(pred_Y))
    f1_score_micro = f1_score(
        test_Y, pred_Y, average='micro', labels=np.unique(pred_Y))
    rocauc_score = roc_auc_score(
        test_Y, clf.predict_proba(test_X), multi_class='ovr')
    return {
        'accuracy': accuracy,
        'macro_avg': macro_avg,
        'f1_score_macro': f1_score_macro,
        'micro_avg': micro_avg,
        'f1_score_micro': f1_score_micro,
        'roc_auc_score': rocauc_score
    }


def binarize(dataset, label):
    res = []
    for data in dataset:
        if int(data) == label:
            res.append(1)
        else:
            res.append(-1)
    return res


def get_roc_auc_curve(clf, X_train, y_train, X_test, y_test, options):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    n_classes = len(np.unique(y_train))
    is_svm = options['is_svm']
    classifier = OneVsRestClassifier(clf)
    if is_svm:
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        binarized_y_test = binarize(y_test, i)
        fpr[i], tpr[i], _ = roc_curve(binarized_y_test, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return (fpr, tpr, roc_auc)


def visualize_roc_auc_curve(title, fpr, tpr, roc_auc, n_classes):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(
            roc_auc["macro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue",
                    "darkviolet", "azure", "burlywood",
                    "coral", "darksalmon", "darkseagreen"])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(
                i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
