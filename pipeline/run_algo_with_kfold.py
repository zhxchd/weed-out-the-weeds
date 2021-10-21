from sklearn.model_selection import KFold
from sklearn import neighbors, tree

def run_knn(train_X, train_Y, test_X, test_Y, options):
  n_neighbors = options['n_neighbors']
  clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
  clf.fit(train_X, train_Y)
  accuracy = clf.score(test_X, test_Y)
  return accuracy

def run_decision_tree(train_X, train_Y, test_X, test_Y, options):
  depth = options['depth']
  clf = tree.DecisionTreeClassifier(max_depth=depth)
  clf = clf.fit(train_X, train_Y)
  accuracy = clf.score(test_X, test_Y)
  return accuracy

algos = { 'knn': run_knn, 'decision_tree': run_decision_tree }

def kfold_cross_validation(k, train_X, train_Y, algo, options):
  print(f'Running {k}-fold cross validation for {algo} with {str(options)}')
  kf = KFold(n_splits=k)

  accuracies = []

  for train_index, test_index in kf.split(train_X):
    cf_train_X = [train_X[index] for index in train_index]
    cf_train_Y = [train_Y[index] for index in train_index]
    cf_test_X = [train_X[index] for index in test_index]
    cf_test_Y = [train_Y[index] for index in test_index]
    accuracy = algos[algo](cf_train_X, cf_train_Y, cf_test_X, cf_test_Y, options)
    accuracies.append(accuracy)
    print(f'Obtained split accuracy of: {str(accuracy)}')

  assert(len(accuracies) == k)
  average_accuracy = sum(accuracies) / k
  print(f'Completed {k}-fold cross validation for {algo} with {str(options)}')
  print(f'Obtained average accuracy of: {average_accuracy}')
  return average_accuracy
  