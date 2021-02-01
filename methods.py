import utilities as ut
import numpy as np
from sklearn.svm import SVC
import pylab as plt
from sklearn import model_selection
from sklearn import metrics
# sklearn 回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


def run_method(method, X, y, n_clfs=6, fs_functions=None, score_name="auc"):
    if method == "forward_selection":
        """
        Forward selection using weighted svm w.r.t greedy, pearson and fisher
        
        Description in section 5.1 - Results in Fig. 9
        """
        w_svm = SVC(class_weight='balanced', probability=True)

        for fs in fs_functions:
            print("FEATURE SELECTION: %s\n" % fs)

            # GET FEATURES RANK
            if fs in ["pearson", "fisher"]:
                print("Ranking features using %s ..." % fs)
                ft_ranks, scores = ut.rank_features(np.array(X), y, corr=fs)

                scores, selected_features = ut.compute_feature_curve(w_svm, X, y,
                                                                     ft_ranks=ft_ranks,
                                                                     step_size=3,
                                                                     score_name=score_name)

            elif fs == "greedy":
                # Greedy selection with auc
                scores, selected_features = ut.greedy_selection(w_svm, X, y, score_name=score_name)

            plt.plot(selected_features, scores, label=fs)

        plt.xlabel("Number of retained features")
        return scores  # 后加的

    elif method == "ensemble_svm":
        """
        Description in section 5.3 - Results in Fig. 10
        """
        clfs = []
        for c in [1, 10, 100, 500, 1000]:
            for w in [{1: 5}, {1: 10}, {1: 15}, {1: 20}, {1: 25}]:
                clfs += [SVC(probability=True, C=c, class_weight=w)]

        (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
        plt.plot(x_values, scores, label="weighted-svm ensemble")


    elif method == "ensemble_heter":
        """    随机森林
        Description in section 5.4 - Results in Fig. 11
        """
        clfs = [SVC(probability=True), MultinomialNB(alpha=0.001),
                BernoulliNB(alpha=0.001), RandomForestClassifier(n_estimators=20),
                GradientBoostingClassifier(n_estimators=300),
                SGDClassifier(alpha=.0001, loss='log', n_iter_no_change=50,
                              penalty="elasticnet"), LogisticRegression(penalty='l2')]

        (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
        print("best auc is :", max(scores))  # 当做目标2的返回值好了
        # print("helllllllo")

        # print(scores)
        plt.plot(x_values, scores, label="heterogenuous ensemble")
        return scores  # 后加的
        print(max(scores))  # 当做目标2的返回值好了

    elif method == "naive":
        """
        直接用多目标来跑
        """
        # clfs = [RandomForestClassifier(n_estimators=20)]
        clfs = [KNeighborsClassifier()]  #KNN
        n_clfs = len(clfs)
        (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
        print("best auc is :", max(scores))  # 当做目标2的返回值好了
        # print("helllllllo")

        # print(scores)
        plt.plot(x_values, scores, label="heterogenuous ensemble")
        return scores  # 后加的
        print(max(scores))  # 当做目标2的返回值好了

        # w_svm = SVC(class_weight='balanced', probability=True)
        # ft_ranks = X
        # scores, selected_features = ut.compute_feature_curve(w_svm, X, y,
        #                                                      ft_ranks=ft_ranks,
        #                                                      step_size=X.shape[1]+1,
        #                                                      score_name=score_name)

        # score_function = score_name
        # score = np.mean(cross_val_score(
        #     clf, X[:, selected_features + [j]], y, cv=4,
        #     scoring=score_function))
        # return scores

    else:
        print("%s does not exist..." % method)
        raise


#### ENSEMBLE FORWARD PASS
def ensemble_forward_pass(clfs, X, y, n_clfs=None):
    if n_clfs == None:
        n_clfs = len(clfs)

    clf_list = ut.ensemble_clfs(clfs)
    auc_scores = np.zeros(n_clfs)

    for i in range(n_clfs):
        skf = model_selection.StratifiedKFold(n_splits=4)

        # CROSS VALIDATE
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf_list.fit(X_train, y_train, i)
            y_pred = clf_list.predict(X_test)

            scores += [metrics.roc_auc_score(y_test, y_pred)]

        auc_scores[i] = np.mean(scores)
        print("Score: %.3f, n_clfs: %d" % (auc_scores[i], i + 1))

    return auc_scores, np.arange(n_clfs) + 1
