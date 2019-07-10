#!/bin/sh

/space/hadoop/lib/python/bin/python3 train_typeDiabetes_classifier.py \
  --pathWordEmbedding "/space/tmp/FastText_embedding_20190703/ft_wordembeddings_dim300_minCount5_URL-User-toConstant_iter10_20190703" \
  --typeWordEmbedding "ft" \
  --pathTrainingSet "/space/Work/spark/Tweet-Classification-Diabetes-Distress/data/ManualLabel_TypeDiabetes_Sexe.csv" \
  --modelAlgo "SVC" \
  --parameterGrid '{"model__C":[1.0, 0.1, 0.01], "model__tol":[1e-1, 1e-2], "model__kernel":["linear", "rbf", "poly"], "smote__k_neighbors":[3,4,5], "union__transformer_weights": [{"tweet":1, "userDesc":0.8}, {"tweet":1, "userDesc":0.5}, {"tweet":1, "userDesc":0.3}]}' \
#  --savePathTrainedModel "bestModel_Jokes_20190707.model" \

#  --parameterGrid '{"model__C":10.0, "model__tol":1e-1, "model__gamma":0.1, "model__kernel":"poly"}' \
