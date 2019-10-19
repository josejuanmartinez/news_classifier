# KAGGLE NEWS CLASSIFIER
A Huffpost (from KAGGLE) News Classifier using different statistical and neural algorithms


![Kaggle Huffpost News Classifier Example](https://drive.google.com/uc?export=view&id=1pyGbFRoVmDagmbvCJpngj-ZBjJLe_04w "Kaggle Huffpost News example")

Note: Doc2Vec models ARE NOT IN THIS REPOSITORY, download them from https://github.com/jhlau/doc2vec and install them in /embeddings folder

Example:
```
python Classifier.py --test_size 0.25 --algo 'Naive Bayes' 'Decision Tree' 'Adaboost' 'Support Vector Machine' 'Random Forest' 'Gradient Descent' --min_feat_size 1 --cleanse --embeddings 'enwiki_dbow'

Usage: Classifier.py [-h] [--cleanse] [--features FEATURES [FEATURES ...]]
                     [--algo ALGO [ALGO ...]] [--min_feat_size MIN_COMB_SIZE]
                     [--embeddings EMBEDDINGS] [--test_size TEST_SIZE]

Runs Classifier

optional arguments:
  -h, --help            show this help message and exit
  --cleanse
  --features FEATURES (Example: --features 'd2v' 'link' 'authors' 'headline' 'short_description')
  --algo ALGO (Example: --algo 'Naive Bayes' 'Decision Tree' 'Adaboost' 'Support Vector Machine' 'Random Forest' 'Gradient Descent')
  --min_feat_size MIN_COMB_SIZE (Example: 1 for all combinations)
  --embeddings EMBEDDINGS (Example: 'apnews_dbow' OR 'enwiki_dbow' OR ...)
  --test_size TEST_SIZE (Example: 0.25)
```
