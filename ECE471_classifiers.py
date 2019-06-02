# ECE471_classifiers.py
# ECE471 Dr.Qi
# written by Noah Caldwell
# 5/6/19
# Used in final project.

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import VotingClassifier

import util
import operator



def MPP_Discriminant(training, testing, case):
    if case != 1 and case != 2 and case != 3:
        raise "Invalid case for MPP discriminant function!"
    
    mpp_classifier = mpp(case)
    training_points = training.T[:-1].T
    testing_points = testing.T[:-1].T
    point_classes = training.T[-1].astype(int)

    mpp_classifier.fit(training_points, point_classes)

    guesses = mpp_classifier.predict(testing_points)
    #print(guesses)
    return guesses

def kNN(training, testing, k):
    knn_classifier = KNeighborsClassifier(n_neighbors=k, p=3)
    # Take the actual class off of the points
    training_points = training.T[:-1].T
    testing_points = testing.T[:-1].T
    point_classes = training.T[-1]
    knn_classifier.fit(training_points, point_classes)
    guesses = knn_classifier.predict(testing_points)
    return guesses

def decision_tree(training, testing, mdepth):
    tree = DecisionTreeClassifier(max_depth=mdepth)
    # Take the actual class off of the points
    training_points = training.T[:-1].T
    testing_points = testing.T[:-1].T
    point_classes = training.T[-1]
    tree.fit(training_points, point_classes)

    guesses = tree.predict(testing_points)
    return guesses

def NN(training, testing, hidden_nodes, hidden_layers=1):
    layers = tuple([hidden_nodes]*hidden_layers)
    mlp = MLPClassifier(validation_fraction=0.111111, hidden_layer_sizes=layers,max_iter=10000)
    # Take the actual class off of the points
    training_points = training.T[:-1].T
    testing_points = testing.T[:-1].T
    point_classes = training.T[-1]
    mlp.fit(training_points, point_classes)

    guesses = mlp.predict(testing_points)
    return guesses


# Our final, combined classifier using classifier fusion. Meant to be used with:
#                Red Wine:                   White Wine:
#   columns:    1, 5, 6, 9, 10, 11          1, 3, 4, 7, 10, 11
#   PCA:        m = 5                       m = 5
#   kNN:        k = 77                      k = 87
#   DT:         max_depth = 5               max_depth = 5
#   MLP:        hidden_nodes = 11           hidden_nodes = 11
#
def combined_classifier(training, testing, wine_color, weighting):
    k = 77 if (wine_color == "red") else 87
    maxdepth = 5 if (wine_color == "red") else 6
    hidden_nodes = 11

    training_points = training.T[:-1].T
    testing_points = testing.T[:-1].T
    point_classes = training.T[-1]
    
    mpp_c1 = mpp(1)
    mpp_c2 = mpp(2)
    mpp_c3 = mpp(3)
    knn = KNeighborsClassifier(n_neighbors=k)
    dt = DecisionTreeClassifier(max_depth=maxdepth)
    mlp = MLPClassifier(validation_fraction=0.111111, hidden_layer_sizes=(11,),max_iter=10000)

    fused = VotingClassifier(estimators=[('1mpp1', mpp_c1), ('2mpp2', mpp_c2), ('3mpp3', mpp_c3), ('4knn', knn), ('5dt', dt), ('6mlp', mlp)], voting='hard', weights=weighting)

    fused.fit(training_points, point_classes)
    guesses = fused.predict(testing_points)
    return guesses





# # # # # # # # # # # # # # # # # # # # # # # 
# MPP Discriminant Function Implementation  # 
# # # # # # # # # # # # # # # # # # # # # # # 

class mpp:
    def __init__(self, case=1):
        # init prior probability, equal distribution
        # self.classn = len(self.classes)
        # self.pw = np.full(self.classn, 1/self.classn)

        # self.covs, self.means, self.covavg, self.varavg = \
        #     self.train(self.train_data, self.classes)
        self.case_ = case
        self.pw_ = {}


    def fit(self, Tr, y):
        # derive the model 
        self.covs_, self.means_ = {}, {}
        self.covsum_ = None

        self.classes_ = np.unique(y).astype(int)     # get unique labels as dictionary items
        self.classn_ = np.max(self.classes_) + 1

        for c in self.classes_:
            arr = Tr[y == c]
            self.covs_[c] = np.cov(np.transpose(arr))
            self.means_[c] = np.mean(arr, axis=0)  # mean along rows
            self.pw_[c] = float(len(arr))/float(len(Tr))
            if self.covsum_ is None:
                self.covsum_ = self.covs_[c]
            else:
                self.covsum_ += self.covs_[c]

        # used by case II
        self.covavg_ = self.covsum_ / self.classn_

        # used by case I
        self.varavg_ = np.sum(np.diagonal(self.covavg_)) / len(self.classes_)
    
    def get_params(self, deep=True):
        return {'case': self.case_}

    def predict(self, T):
        # eval all data 
        y = []
        disc = {}
        nr, _ = T.shape

        if self.pw_ is None:
            self.pw_ = np.full(self.classn_, 1 / self.classn_)

        for i in range(nr):
            for c in self.classes_:
                try:
                    if self.case_ == 1:
                        edist2 = util.euc2(self.means_[c], T[i])
                        disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
                    elif self.case_ == 2: 
                        mdist2 = util.mah2(self.means_[c], T[i], self.covavg_)
                        disc[c] = -mdist2 / 2 + np.log(self.pw_[c])
                    elif self.case_ == 3:
                        mdist2 = util.mah2(self.means_[c], T[i], self.covs_[c])
                        disc[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                                    + np.log(self.pw_[c])
                    else:
                        print("Can only handle case numbers 1, 2, 3.")
                        sys.exit(1)
                except:
                    continue
            #y.append(disc.argmax())
            y.append(max(disc.items(), key=operator.itemgetter(1))[0])
            
        return y

    def predict_proba(self, T):
        # eval all data 
        y = []
        disc = {}
        nr, _ = T.shape
        probtable = []

        for i in range(nr):
            probs = np.full((int(self.classn_,)), np.finfo('float64').min)
            for c in self.classes_:
                c = int(c)
                try:
                    if self.case_ == 1:
                        edist2 = util.euc2(self.means_[c], T[i])
                        probs[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
                    elif self.case_ == 2: 
                        mdist2 = util.mah2(self.means_[c], T[i], self.covavg_)
                        probs[c] = -mdist2 / 2 + np.log(self.pw_[c])
                    elif self.case_ == 3:
                        mdist2 = util.mah2(self.means_[c], T[i], self.covs_[c])
                        probs[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                                    + np.log(self.pw_[c])
                    else:
                        print("Can only handle case numbers 1, 2, 3.")
                        sys.exit(1)
                except:
                    continue

            probs = np.exp(probs)
            probs = np.divide(probs, np.sum(probs))
            probtable.append(probs)
            
        return np.array(probtable)
