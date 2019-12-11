# Activity-Prediction-for-Chemical-Compounds
Course Project for ID2214 HT19-1 Programming for Data Science in KTH Royal Institute of Technology

**Introduction:**

We applied traditional machine learning models such as Logistic regression, Random Forest, LightGBM and also deep learning methods MolCNN which combines Fine tune CNN with pretrained Mol2vec to the problem and employed cross validation to ensure the validity and avoid overfitting, and use AUC as the metric of evaluation. 

We presents the whole modeling process with an emphasis on feature engineering, model selection and model explanation. The final results comes from using Mol2vec features and LightGBM model.

Some detailed could be found in our presentation slides

[1] Jaeger, Sabrina, Fulle, Simone, and Turk, Samo. "Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition." Journal of Chemical Information & Modeling :acs.jcim.7b00616.


**Requirements Package:**

* Traditional machine learning methods:

  1. pickle

  2. sklearn

  3. pandas

  4. numpy

  5. rdkit

  6. mol2vec (https://github.com/samoturk/mol2vec)

  7. gensim

  8. lightgbm

* MolCNN

  1. torch

  2. mol2vec

  3. gensim

  4. pandas

  5. numpy
