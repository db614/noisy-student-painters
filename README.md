# Noisy Student Training Implementation

This is an implementation of the ["Noisy Student"](https://arxiv.org/abs/1911.04252) paper, in which a teacher model is trained on a small quantity of labeled data and is then used to produce pseudolabels for a much larger quantity of unlabeled data. The pseduolabeled data is then perturbed, and used to train a student model, which in turn produces new labels for the data. This process of repeated learning, labeling, and re-learning of perturbed data can be repeated to increase the overall accuracy of the model and improve its performance with new data. In the paper an 88.4% top-1 accuracy accuracy was achieved on ImageNet as well as improvements in robustness benchmarks.

I have used the [Painter by Numbers](https://www.kaggle.com/competitions/painter-by-numbers/data) dataset from Kaggle in this implementation.

## Usage

Once the kaggle data is downloaded, the annotations file can be converted to suitably formatted csv files using the "data_prep.ipynb" Jupyter Notebook. Training, validation, and scoring of the model can be run from the "Run.ipynb" Jupyter Notebook.
