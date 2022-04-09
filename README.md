# How to run：
`MOEAD.py`
# Introduction
Software defect prediction is a technology that provides decision support for software testing resource allocation by identifying suspicious defect modules in advance for the purpose of improving software quality. But the performance of the software defect prediction model depends on the quality of the software features considered. Redundant and irrelevant features may reduce the performance of the model, which requires feature selection methods to identify and remove such features. This paper conducts in-depth research on feature selection in software defect prediction. The main research work is as follows:

(1) This paper transforms the feature selection problem of software defect prediction into a multi-objective optimization problem. Combining the practical significance of feature engineering, after comprehensively analyzing model running time, feature subset size, classification machine learning algorithm performance, model performance, etc., the optimization goal is selected to minimize the number of selected features and maximize the software defect prediction model performance.

(2) The multi-objective optimization feature selection method in this paper adopts the high-dimensional multi-objective optimization MOEA/D algorithm based on the decomposition strategy to construct, which is called MOO-SDPFS (Feature selection for software defect prediction based on multi-objective optimization). Subsequently, we applied the above method to the NASA MDP defect database (selected CM1, KC3, MC1, MW1, PC2, PC4 and PC5 eight actual project data sets) and PROMISE defect database (selected ant-1.7, camel-1.6, jedit-4.3 and xerces-2.0 four actual project data sets) for comprehensive research. The experimental results show that the MOO-SDPFS method can effectively deal with the feature selection problem in software defect prediction within acceptable time expenditure. In addition, this paper continues to explore the pros and cons of multi-objective optimization feature selection methods based on filtering, wrapping, and embedded. Finally, an empirical analysis is conducted based on the experimental results to discuss the positive effects of multi-objective optimization feature selection in software defect prediction.






# Based :
# 1、software_defect_prediction
Software defect prediction using ensemble learning

This contains the code for generating the results in the paper:
http://www.sciencedirect.com/science/article/pii/S0950584914001591

Run `main.py` to experiment with different datasets and models.
# 2、MOEA/D
本代码是对MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition论文中算法编程实现。

MOEAD算法论文大致介绍详细，这个中文的帖子也不错：https://blog.csdn.net/sinat_33231573/article/details/80271801
