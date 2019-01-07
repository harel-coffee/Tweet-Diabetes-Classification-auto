#!/bin/sh

/space/hadoop/lib/python/bin/python3 train_classifier.py \
  --mode "local" \
  --pathWordEmbedding "/space/Work/spark/FastText_model/ft_wordembeddings_09112018.model" \
  --typeWordEmbedding "ft" \
  --pathTrainingSet "/space/Work/spark/manually_labeled_users_instVSpers_MoreInstTweets_30072018.csv" \
  --columnNameLabel "personal (0=no, 1=yes)" \
  --columnNameTextData "tweet" \
  --parameterGrid '{"model__kernel" : ["linear", "poly", "rbf"], "model__C" : [30.0, 25.0, 20.0, 15.0, 12.0, 10.0], "model__tol" : [1e-1, 1e-2, 1e-3], "model__gamma" : ["auto", 0.01, 0.1, 1.0]}' \
  --modelAlgo "SVC" \


#  --parameterGrid '{"model__C":[10.0, 1.0, 0.1, 0.01], "model__tol":[1e-10, 1e-9, 1e-8, 1e-7, 1e-6], "model__solver":["liblinear"]}' \
#  --modelAlgo "logReg"



  # parameters_ft = {
  #               # param for MultinomialNB
  #               #'model__alpha': (10, 5, 1, 0.5, 0.1),
  #
  #               # param for LogisticRegression
  #               #'model__C' : [5, 3, 1.0, 0.8, 0.5, 0.1],
  #               #'model__tol' : [1e-10, 1e-9],
  #
  #               # param for SVC
  #               'model__kernel' : ["linear", "poly", "rbf"],
  #               'model__C' : [ 12.0, 10.0, 8.0, 6.0, 4.0, 1.0],
  #               'model__tol' : [1e-1, 1e-2, 1e-3],
  #
  #               # param for RandomForestClassifier
  #               #'model__n_estimators' : [70, 80, 100, 120],
  #               #'model__criterion' : ['gini', 'entropy'],
  #               #'model__max_features' : ['auto', 'log2'],
  #               #'model__max_depth' : [ 8, 10, 20]
  #
  #               # param for XGBoost Best: 0.910828 using {'model__learning_rate': 0.05, 'model__reg_alpha': 0, 'model__max_depth': 3, 'model__reg_lambda': 1.5, 'model__n_estimators': 300}
  #               #'model__max_depth' : [3,4],
  #               #'model__learning_rate' : [0.03, 0.05, 0.07],
  #               #'model__booster' : ["gblinear"], #["gbtree", "gblinear", "dart"],
  #               #'model__gamma' : [0, 0.01],
  #               #'model__n_estimators' : [200, 300, 400, 500],
  #               #'model__reg_alpha' : [0, 0.1],
  #               #'model__reg_lambda' : [0.5, 1.0, 1.5]
  #
  #               # param for Multi layer perceptron
  #               #'model__hidden_layer_sizes' : [(64,64), (64,32)],#[(64), (64, 32), (32, 32)],
  #               #'model__activation' : ['relu', 'tanh', 'logistic'],
  #               #'model__solver' : ['adam', 'sgd'],
  #               #'model__learning_rate' : ['constant', 'invscaling'],
  #               #'model__tol' : [1e-2, 1e-3],#[1e-2, 1e-3, 1e-4],
  #
  #               #'model__hidden_layer_sizes' :  [(64), (32), (16,16)],#[(64), (16, 16), (32, 16)],
  #               #'model__activation' : ['relu'],# ['relu', 'tanh'],
  #               #'model__solver' : ['adam'],#['adam', 'sgd'],
  #               #'model__learning_rate' : ['constant', 'invscaling'],
  #               #'model__tol' : [1e-2, 1e-3, 1e-4],
  #               #'model__alpha' : [ 1e-4, 1e-5, 1e-6],
  #               #'model__max_iter' : [200, 300],
  #               #'model__beta_1' : [0.990, 0.999],
  #               #'model__beta_2' : [1e-7, 1e-8, 1e-9]
  #
  # }
