# KAGGLE NEWS CLASSIFIER
A Huffpost (from KAGGLE) News Classifier using different statistical and neural algorithms

Note: Doc2Vec models ARE NOT IN THIS REPOSITORY, download them from https://github.com/jhlau/doc2vec and install them in /embeddings folder
 
```
Usage: Classifier.py [-h] [--cleanse] [--features FEATURES [FEATURES ...]]
                     [--algo ALGO [ALGO ...]] [--min_feat_size MIN_COMB_SIZE]
                     [--embeddings EMBEDDINGS] [--test_size TEST_SIZE]

Runs Classifier

optional arguments:
  -h, --help            show this help message and exit
  --cleanse
  --features FEATURES (Example: "d2v, link, authors, headline, short_description")
  --algo ALGO (Example: "Naive Bayes, Decision Tree, Adaboost, Support Vector Machine, Random Forest, Gradient Descent")
  --min_feat_size MIN_COMB_SIZE (Example: 1 for all combinations)
  --embeddings EMBEDDINGS (Example: 'apnews_dbow', 'enwiki_dbow')
  --test_size TEST_SIZE (Example: 0.25)
```
