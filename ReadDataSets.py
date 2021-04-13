import utilities as ut
import time
import argparse

##### 2. ------- RUN TRANING METHOD
# method = "forward_selection"
method = "ensemble_heter"  # 集成学习
# method = "naive"  # naive方法
# subMethod = "W_SVM"
# method = "ensemble_svm"  # 集成的SVM

# fs_functions = ["pearson"]
fs_functions = ["greedy"]
score_name = "auc"

# 集成学习弱分类器个数
n_clfs = 3
# # 弱分类器为 clfs = [SVC(probability=True), MultinomialNB(alpha=0.001),
#                 BernoulliNB(alpha=0.001), RandomForestClassifier(n_estimators=20),
#                 GradientBoostingClassifier(n_estimators=300),
#                 SGDClassifier(alpha=.0001, loss='log', n_iter_no_change=50,
#                               penalty="elasticnet"), LogisticRegression(penalty='l2')]

start = time.time()
# dataset_name = "KC3"
# dataset_name = "CM1"
# dataset_name = "camelOSA"
dataset_name = "MC1"
# dataset_name = "MC2"
# dataset_name = "MW1"
# dataset_name = "PC2"
# dataset_name = "PC4"
# dataset_name = "PC5"

# ###########
# moead = MOEAD()

XX, y, ft_names = ut.read_dataset("D:/PycharmProjects/software_defect_prediction-master/datasets/",
                                  dataset_name=dataset_name)
