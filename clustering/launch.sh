#!/bin/sh

/space/hadoop/lib/python/bin/python3 kmeans.py \
	--filename "/space/tmp/matching-tweets_diab_noRT-noBots_personal_noJokes_locationUS_geoCityCodeNotNull_emotions.parquet" \
	-lfc "id, text, user_name" \
	--wordEmbedding "/space/tmp/FastText_embedding_20190703/ft_wordembeddings_dim300_minCount5_URL-User-toConstant_iter10_20190703" \
	--dataColumnName "text" \
	--maxIterations 3 \
	--Ncluster 5 \
	--savePath "/space/tmp/matching-tweets_diab_noRT-noBots_personal_noJokes_locationUS_geoCityCodeNotNull_emotions_clustered.parquet"
