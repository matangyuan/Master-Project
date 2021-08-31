import argparse
from train import *
from prepare import *
from models import *
from transformers import RobertaForSequenceClassification, BertForSequenceClassification, XLNetForSequenceClassification
from torchcrf import CRF
from predict import *
from eval import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='RoBERTa-BiLSTM-CRF', type=str,
                        help='Choose the model from: BERT, BERT-BiLSTM-CRF, RoBERTa, RoBERTa-BiLSTM-CRF')
    parser.add_argument('--training_data_path', default='Training dataset/', type=str, help='Specify the training data folder')
    parser.add_argument('--testing_data_path', default='Testing dataset/', type=str, help='Specify the testing data folder')
    parser.add_argument('--save_path', default='modesl/', type=str,
                        help='Specify the folder to store the models and predictions')

    parser.add_argument('--max_length', default=256, type=int, help='The maximal length for the input sentence.')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1.75e-5, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--epochs', default=4, type=int)
    args = parser.parse_args()

    # Identify the device in use
    if torch.cuda.is_available():

        # Use the GPU if possible
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('The GPU in use:', torch.cuda.get_device_name(0))

    # Use CPU then:
    else:
        print('CPU is used in this notebook.')
        device = torch.device("cpu")

    data, test_data = load_data(args)
    sentences, labels, test_sentences, test_labels, test_df = create_df(data, test_data, args)

    # firstly use tokenization tools to tokenize the sentence and
    if args.model == 'BERT-BiLSTM-CRF':
        train_dataloader, prediction_dataloader, tokenizer = tokenize_BERT_sentence(sentences, labels, test_sentences,
                                                                                    test_labels, args)

        model = BertSSC.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        model.cuda()
    elif args.model == 'BERT':
        train_dataloader, prediction_dataloader, tokenizer = tokenize_BERT_sentence(sentences, labels, test_sentences,
                                                                                    test_labels, args)

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=7,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        # Tell pytorch to run this model on the GPU.
        model.cuda()
    elif args.model == 'RoBERTa':
        train_dataloader, prediction_dataloader, tokenizer = tokenize_RoBERTa_sentence(sentences, labels, test_sentences,
                                                                                     test_labels, args)
        # Load RobertaForSequenceClassification, the pretrained RoBERTA model with a single
        # linear classification layer on top.
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",  # Use the 12-layer RoBERTA model, with an uncased vocab.
            num_labels=7,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.cuda()
    elif args.model == 'RoBERTa-BiLSTM-CRF':
        train_dataloader, prediction_dataloader, tokenizer = tokenize_RoBERTa_sentence(sentences, labels, test_sentences,
                                                                                     test_labels, args)
        model = RoBertaSSC.from_pretrained(
            "roberta-base",  # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        model.cuda()
    elif args.model == 'XLNet':
        train_dataloader, prediction_dataloader, tokenizer = tokenize_XLNet_sentence(sentences, labels, test_sentences,
                                                                                     test_labels, args)
        # Load RobertaForSequenceClassification, the pretrained RoBERTA model with a single
        # linear classification layer on top.
        model = XLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',  # Use the 12-layer RoBERTA model, with an uncased vocab.
            num_labels=7,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.cuda()
    else:
        print('no such models.')
        return

    model = train(train_dataloader, model, device, args)
    predict(prediction_dataloader, test_df, model, device, tokenizer, args)
    eval(args)


if __name__ == '__main__':
    main()
