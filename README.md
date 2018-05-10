# sentiment-analysis-reviews

## Introduction
During this project, I implemented the Naive Bayes algorithm with smoothing for the task of sentiment classification based on YELP reviews identifying positive and negative product reviews.

## Dataset
The dataset consists of sentences with sentiment labels (1 for positive and 0 for negative).Each line is a list of space separated words, which essentially a sentence, followed by a tab character, and then followed by the label. Here is a snippet from the yelp dataset:
```
Crust is not good. 0 

Best burger ever had! 1
```

## Implementation
### Naive Bayes 
Implement a Naive Bayes classifier to predict sentiment labels of sentences from scratch.

### Document to a vector
I convert a document to a vector by checking whether a word appears in the document. Each word corresponds to a feature. If a word j appear in the document, then the j-th feature of the document is 1; otherwise its j-th feature is 0. In this way, each document is converted to a vector with length of the vocabulary size.

## Python files
[yelp_labelled.txt](https://github.com/yding06/sentiment-analysis-reviews/blob/master/yelp_labelled.txt): contains all reviews from Yelp.com

[preprocessing.py](https://github.com/yding06/sentiment-analysis-reviews/blob/master/preprocessing.py): preprocess data from [yelp_labelled.txt](https://github.com/yding06/sentiment-analysis-reviews/blob/master/yelp_labelled.txt): contains all reviews from Yelp.com and return [yelp_features_1800.csv](https://github.com/yding06/sentiment-analysis-reviews/blob/master/yelp_%20features_1800.csv) and [yelp_labels.csv](https://github.com/yding06/sentiment-analysis-reviews/blob/master/yelp_labels.csv)

[naive-bayes-yelp.py](https://github.com/yding06/sentiment-analysis-reviews/blob/master/naive-bayes-yelp.py): a Naive bayes model based on the returned files above
