import pandas as pd
import glob
import sklearn
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score


def calc_classification_metrics(y_true, y_predicted, labels):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='micro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='weighted')
    per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average=None, labels=labels)

    acc = accuracy_score(y_true, y_predicted)

    class_report = classification_report(y_true, y_predicted, digits=4)
    confusion_abs = confusion_matrix(y_true, y_predicted, labels=labels)
    # normalize confusion matrix
    confusion = np.around(confusion_abs.astype('float') / confusion_abs.sum(axis=1)[:, np.newaxis] * 100, 2)
    return {"acc": acc,
            "macro-f1": macro_f1,
            "macro-precision": macro_precision,
            "macro-recall": macro_recall,
            "micro-f1": micro_f1,
            "micro-precision": micro_precision,
            "micro-recall": micro_recall,
            "weighted-f1": weighted_f1,
            "weighted-precision": weighted_precision,
            "weighted-recall": weighted_recall,
            "labels": labels,
            "per-label-f1": per_label_f1.tolist(),
            "per-label-precision": per_label_precision.tolist(),
            "per-label-recall": per_label_recall.tolist(),
            "confusion_abs": confusion_abs.tolist()
            }, \
           confusion.tolist(), \
           class_report


def eval(args):
    # read data from golden samples
    # DATA_DIC = r"/content/drive/My Drive/AILA Datasets/"
    # os.makedirs(DATA_DIC, exist_ok=True)
    gold_data = pd.read_csv('task2-labels.csv', sep=',')

    # read results
    predict_data = pd.read_csv(
         args.save_path + args.model_name + '.txt', sep='\t', header=None)
    predict_data = predict_data[[0, 1]]
    predict_data.columns = ['sent_id', 'label']

    data = predict_data.merge(gold_data, on='sent_id')

    data_pred = data.label_x
    data_gold = data.label_y

    labeldict = {"Facts": 0, "Ruling by Lower Court": 1, "Argument": 2, "Statute": 3, "Precedent": 4,
                 "Ratio of the decision": 5, "Ruling by Present Court": 6}

    y_pred = data_pred.map({"Facts": 0, "Ruling by Lower Court": 1, "Argument": 2, "Statute": 3, "Precedent": 4,
                            "Ratio of the decision": 5, "Ruling by Present Court": 6})
    y_gold = data_gold.map({"Facts": 0, "Ruling by Lower Court": 1, "Argument": 2, "Statute": 3, "Precedent": 4,
                            "Ratio of the decision": 5, "Ruling by Present Court": 6})

    labels = list(range(0, 7))

    result = calc_classification_metrics(y_gold, y_pred, labels)
    print(result)

    file1 = open(
        args.save_path+args.model_name+"_FINALRESULT.txt", "a")
    file1.write('acc:')
    file1.write('\t')
    file1.write(str(result[0]['acc']))
    file1.write('\n')
    file1.write('macro-f1:')
    file1.write('\t')
    file1.write(str(result[0]['macro-f1']))
    file1.write('\n')
    file1.write('macro-precision:')
    file1.write('\t')
    file1.write(str(result[0]['macro-precision']))
    file1.write('\n')
    file1.write('macro-recall:')
    file1.write('\t')
    file1.write(str(result[0]['macro-recall']))
    file1.write('\n')
    file1.write('confusion matrix:')
    file1.write('\t')
    file1.write(str(result[1]))
    file1.write('\n')
    file1.close()


