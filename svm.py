import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings

warnings.simplefilter("ignore")
    
# import iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features
y = iris.target


# cut 2 times the dataset in order to have
# train, validation, test sets in proportion 5:2:3
X_train, other1, y_train, other2 = train_test_split(
    X, y, test_size=0.5, random_state=1)

X_val, X_test, y_val, y_test = train_test_split(
    other1, other2, test_size=0.6, random_state=1)

# generate lists of parameters for GridSearch
C_list = [C for C in (0.0001*(10**p) for p in range(1,8))]
gamma_l = [C for C in [0.1**n for n in range(1,8)]]

################################## LINEAR SVM ################################
# 
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
# C is SVM regularization parameter

idx  = 0
scores = np.empty([7,2])
plt.title('SVC with linear kernel.')

for C in C_list:
    # initialize linear SVM with C
    clf = svm.SVC(kernel='linear', C=C, gamma='auto')

    # training with training set
    clf.fit(X_train, y_train)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    # evaluate on validation set
    scr = clf.score(X_val, y_val) * 100 
    scores[idx] = (C,scr) # save C and realtive score
    t = "C = {} Accuracy: {:.2f}%".format(C,scr)
    idx += 1
    plt.subplot(3, 3, idx).set_title(t)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    
plt.subplots_adjust(hspace=0.4)
plt.show()

# get the best C 
C = scores[np.argmax(scores,axis=0)[1]][0]

# evaluate with the above C on the test set
clf = svm.SVC(kernel='linear', C=C, gamma='auto')
clf.fit(X_train, y_train)
scr = clf.score(X_test, y_test) * 100 
print("Best accuracy is {:.2f} with C = {} \n".format(scr,C))



################################## RBF SVM ##################################

idx = 0
scores = np.empty([7,2])
plt.title("SVM with RBF Kernel")
for C in C_list:
    # initialize kernel SVM with C
    clf = svm.SVC(kernel='rbf', C = C, gamma='auto')
    # training with training set
    clf.fit(X_train, y_train)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    # define subplot parameters
    scr = clf.score(X_test, y_test) * 100 
    scores[idx] = (C,scr)
    t = "C = {} Accuracy: {:.2f}%".format(C,scr)
    idx += 1
    plt.subplot(3, 3, idx).set_title(t)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    
plt.subplots_adjust(hspace=0.4)
plt.show()

# get the best C 
C = scores[np.argmax(scores,axis=0)[1]][0]

# evaluate with the above C on the test set
clf = svm.SVC(kernel='rbf', C=C, gamma='auto')
clf.fit(X_train, y_train)
scr = clf.score(X_test, y_test) * 100 
print("Best accuracy is {:.2f} with C = {} \n".format(scr,C))


######################################################################
# ######################### Grid Search ##############################
######################################################################

# Parameter Grid
param_grid = {'C': C_list, 'gamma': gamma_l, 'kernel': ['rbf']}
 
# Make grid search classifier
clf = GridSearchCV(svm.SVC(), param_grid, verbose=0, cv = 3)
 
# Train the classifier
clf.fit(X_train, y_train)
 
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_val, clf.predict(X_val)
print(classification_report(y_true, y_pred))
print()

# plot with the best parameters just found
C = clf.best_params_['C']
gamma = clf.best_params_['gamma']
clf = svm.SVC(kernel='rbf', C = C, gamma= gamma)
# training with training set
clf.fit(X_test, y_test)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

scr = clf.score(X_val, y_val) * 100 
t = "C = {}, gamma = {:.4f} - Accuracy: {:.2f}%".format(C,gamma,scr)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.title(t)    
plt.show()



 # Merge the training and validation split
X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

# Make grid search classifier
clf = GridSearchCV(svm.SVC(), param_grid, verbose=0,cv=5)
 
# Train the classifier
clf.fit(X_train, y_train)

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

scores_mean = clf.cv_results_['mean_test_score']
scores = np.array(scores_mean).reshape(len(C_list), len(gamma_l))

for ind, i in enumerate(C_list):
    plt.plot(gamma_l, scores[ind], label='C: ' + str(i))
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Mean score')
plt.show()
