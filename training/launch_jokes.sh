#!/bin/sh

/space/hadoop/lib/python/bin/python3 train_classifier.py \
  --mode "local" \
  --pathWordEmbedding "/space/tmp/FastText_embedding_20190703/ft_wordembeddings_dim300_minCount5_URL-User-toConstant_iter10_20190703" \
  --typeWordEmbedding "ft" \
  --pathTrainingSet "/space/tmp/ManualLabels_sampleJokes.csv" \
  --columnNameLabel "isJoke" \
  --modelAlgo "SVC" \
  --parameterGrid '{"model__C":[1.0, 0.1, 0.01], "model__tol":[1e-0, 1e-1], "model__solver":["linear", "rbf", "poly"], "smote__k_neighbors":[5,6,7], "union__transformer_weights": [{"tweet":1, "userDesc":0.0}, {"tweet":1, "userDesc":0.1}]}' \
  --savePathTrainedModel "bestModel_Jokes_20190707.model" \
  --scoring "recall"

#  --parameterGrid '{"model__C":10.0, "model__tol":1e-1, "model__gamma":0.1, "model__kernel":"poly"}' \
