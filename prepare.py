import pandas as pd
import torch
import glob

def load_data(args):
    # specify the training data folder
    # files = glob.glob("/content/drive/My Drive/AILA Datasets/Training data/*.txt")
    files = glob.glob(args.training_data_path+'*.txt')

    # specify the testing data folder
    # test_files = glob.glob("/content/drive/My Drive/AILA Datasets/Testing data/*.txt")
    test_files = glob.glob(args.testing_data_path+'*.txt')

    # read training data
    data = []
    for text_file in files:
        with open(text_file, "r") as cfile:
            temp_line = cfile.readlines()
            data.append(temp_line)
    del temp_line
    del files

    # read testing data
    test_data = []
    for text_file in test_files:
        with open(text_file, "r") as cfile:
            temp_line = cfile.readlines()
            test_data.append(temp_line)
    del temp_line
    del test_files

    return data, test_data


def create_df(data, test_data, args):
    # the training data is not well-organized, the better way is to eliminate the space between the labels
    # and remap into corresponding labels
    train_labeldict = {"Facts": 0, "RulingbyLowerCourt": 1, "Argument": 2, "Statute": 3, "Precedent": 4,
                       "Ratioofthedecision": 5, "RulingbyPresentCourt": 6}

    # read training data into dataframe
    # read sequentially from the data folder
    df = pd.DataFrame()
    for text_file in data:
        filedata = {'sentence': [], 'label': []}
        # words_in_sen = line.split()
        # label = words_in_sen[0]
        # temp = line.split('\t', 1)[1]
        # temp = temp.split('\n', 1)[0]
        # filedata['sentence'].append(temp)
        # filedata['label'].append(0)
        # filedata['id'].append(label)
        for line in text_file:
            # split the sentence and eliminate the space between labels
            # remap it based on modified dictionary
            words_in_sen = line.split()
            if words_in_sen[-1] == "Court":
                label = words_in_sen[-4] + words_in_sen[-3] + words_in_sen[-2] + words_in_sen[-1]
                label = train_labeldict[label]
                sen = line.rsplit(' ', 4)[0]
            elif words_in_sen[-1] == "decision":
                label = words_in_sen[-4] + words_in_sen[-3] + words_in_sen[-2] + words_in_sen[-1]
                label = train_labeldict[label]
                sen = line.rsplit(' ', 4)[0]
            else:
                label = words_in_sen[-1]
                label = train_labeldict[label]
                sen = line.rsplit(' ', 1)[0]

            # append the sentence into a temp file
            filedata['sentence'].append(sen)
            filedata['label'].append(label)
        temp = pd.DataFrame(filedata)
        # print(temp)
        # append the temp file to the dataframe
        df = df.append(temp)
    del label, sen, temp, filedata, words_in_sen, data

    # read test data into dataframes
    test_df = pd.DataFrame()
    for text_file in test_data:
        filedata = {'sentence': [], 'label': [], 'id': []}
        for line in text_file:
            words_in_sen = line.split()
            label = words_in_sen[0]
            sen = line.split('\t', 1)[1]
            sen = sen.split('\n', 1)[0]
            filedata['sentence'].append(sen)
            filedata['label'].append(0)
            filedata['id'].append(label)
        temp = pd.DataFrame(filedata)
        test_df = test_df.append(temp)
    del label, sen, temp, filedata, words_in_sen, test_data

    sentences = df.sentence.values
    labels = df.label.values

    test_sentences = test_df.sentence.values
    test_labels = test_df.label.values
    # print(test_sentences)
    return sentences, labels, test_sentences, test_labels, test_df



