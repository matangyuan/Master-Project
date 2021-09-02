# Master-Project

Source code for 'Identification of Rhetorical Roles of Sentences in Legal Judgements Using Transformer-based Approaches.'

## Dependecies
transformers=2.11.0

pytorch-crf=0.7.2

torchtext==0.2.0

numpy==1.13.3

torch==1.9.0

jieba==0.39

scikit_learn==0.21.0

## Dataset
The training dataset we use for this task consists of a set of 50 court case documents judged in the Supreme Court of India (https://sites.google.com/view/aila-2020/dataset-evaluation-plan), where each sentence has been carefully annotated with its rhetorical role by law students. And the desired targets for each sentence are one of the seven rhetorical labels. The seven rhetorical roles are: Facts, Ruling by Lower Court, Argument, Statute, Precedent, Ratio of the decision and Ruling by Present Court. As the dataset is regarded as imbalanced, we introduce multiple solutions for the imbalanced data, such as: Using Dropout layers, Using transformer-based networks etc. 

## Usage
Reproducing the results reported in our paper, please run the code as follows:

`python run.py`
