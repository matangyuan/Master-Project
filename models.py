from transformers import BertConfig, RobertaConfig, XLNetConfig
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from transformers import BertModel,RobertaForSequenceClassification, RobertaModel,RobertaForSequenceClassification, AdamW, RobertaConfig
from torchcrf import CRF


def mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args):
    # In this function, we use tokenizers to tokenize the sentence and its labels
    input_ids = []
    attention_masks = []

    # Loop every sentence
    for sent in sentences:
        # the encoder will tokenize the sentence with pre-trained models
        # [CLS] and [SEP] tokens are added to the sentence
        # the tokens are mapped to its IDs
        # the sentence is padded to the specified length
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,  # [CLS] and [SEP] tokens are added to the sentence
            max_length=args.max_length,  # the sentence is padded to the specified length
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,  # attention mask created
            return_tensors='pt',  # we use pytorch input for later use
        )

        # the encoded sentences are appended to the input_ids
        input_ids.append(encoded_dict['input_ids'])

        # attention mask are appended to its list
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # demonstrate we are doing correctly
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    # For testing data...
    test_input_ids = []
    test_attention_masks = []

    # Loop every test sentence
    for sent in test_sentences:
        # the encoder will tokenize the sentence with pre-trained models
        # [CLS] and [SEP] tokens are added to the sentence
        # the tokens are mapped to its IDs
        # the sentence is padded to the specified length
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # [CLS] and [SEP] tokens are added to the sentence
            max_length=args.max_length,  # the sentence is padded to the specified length
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,  # attention mask created
            return_tensors='pt',  # we use pytorch input for later use
        )

        # the encoded sentences are appended to the input_ids
        test_input_ids.append(encoded_dict['input_ids'])
        # attention mask are appended to its list
        test_attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    test_input_ids = torch.cat(test_input_ids, dim=0)
    test_attention_masks = torch.cat(test_attention_masks, dim=0)
    test_labels = torch.tensor(test_labels)

    # Set the batch size for testing data which is 1
    test_batch_size = 1

    # Create the dataLoader for the testing dataset
    prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=test_batch_size)

    # Creating a TensorDataset which contains input_id, attention mask and labels
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Separate the data into training and validation dataset, validation can be used to determine the correct parameters
    train_size = int(1 * len(dataset))
    val_size = len(dataset) - train_size

    # shuffle the data
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Specify the batch_size parameter for training data. 16 is used here
    batch_size = args.batch_size

    # Separate the data into training and validation dataset
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # use 16 for the batch size
    )

    # validation can be used to determine the correct parameters
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),  # no need to shuffle the validation set anyways
        batch_size=batch_size  # still use 16 for the batch size
    )

    return train_dataloader, prediction_dataloader


def tokenize_BERT_sentence(sentences, labels, test_sentences, test_labels, args):
    configuration = BertConfig()
    # Load the bert-base-cased tokenizer.
    # We still have cased data in our documents so we use cased version of BERT
    print('Loading bert-base-cased tokenizer:')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    # This is a demo of the tokenization process
    print(' Original Sentence: ', sentences[0])
    print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
    print('Token IDs of the Sentence: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    max_len = 0
    # We use this block to identify the longest sentence.
    for sent in sentences:
        # Tokenize the sentence and add special tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Find the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Maximal sentence length: ', max_len)

    train_dataloader, prediction_dataloader = mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args)

    return train_dataloader, prediction_dataloader, tokenizer


def tokenize_RoBERTa_sentence(sentences, labels, test_sentences, test_labels, args):
    configuration = RobertaConfig()
    # Load the RoBERTa tokenizer.
    print('Loading RoBERTa tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    # This is a demo of the tokenization process
    print(' Original Sentence: ', sentences[0])
    print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
    print('Token IDs of the Sentence: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    max_len = 0
    # We use this block to identify the longest sentence.
    for sent in sentences:
        # Tokenize the sentence and add special tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Find the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Maximal sentence length: ', max_len)

    train_dataloader, prediction_dataloader = mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args)

    return train_dataloader, prediction_dataloader, tokenizer

def tokenize_XLNet_sentence(sentences, labels, test_sentences, test_labels, args):
    configuration = XLNetConfig()
    # Load the XLNetTokenizer tokenizer.
    print('Loading XLNetTokenizer tokenizer...')
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    # This is a demo of the tokenization process
    print(' Original Sentence: ', sentences[0])
    print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
    print('Token IDs of the Sentence: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    max_len = 0
    # We use this block to identify the longest sentence.
    for sent in sentences:
        # Tokenize the sentence and add special tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Find the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Maximal sentence length: ', max_len)

    train_dataloader, prediction_dataloader = mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args)

    return train_dataloader, prediction_dataloader, tokenizer


class BertSSC(BertModel):
    def __init__(self, config, lstm_embedding_size = 768, dense_hidden_size = 1536,
                    lstm_dropout_prob = 0.1, num_tags = 7):
        super(BertSSC, self).__init__(config)
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",  # We still use the bert-base-uncased model to capture the semantics
            output_attentions=False,  # No need to returns attentions weights.
            output_hidden_states=False,  # No need to returns all hidden-states.
        )
        # self.dropout = torch.nn.Dropout(0.1)
        self.bilstm = torch.nn.LSTM(
            input_size=lstm_embedding_size,    # We use the BiLSTM layers to output the predictions
            hidden_size=dense_hidden_size // 2,  # specify the hidden_size for each BiLSTM layer
            batch_first=True,
            num_layers=2,   # specify the number of the BiLSTM layer
            dropout=lstm_dropout_prob,
            bidirectional=True  # make sure it is a bidirectional-LSTM layer
        )
        self.classifier = torch.nn.Linear(lstm_embedding_size * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)  # CRF layer is added to improve performance
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)

        bilstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(bilstm_output[:, -1, :])
        logits = logits.unsqueeze(1)

        outputs = (logits,)
        if labels is not None:
            labels = labels.unsqueeze(1)
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        # else: loss = 1
        # returns: (loss), scores
        return outputs


class RoBertaSSC(RobertaModel):
    def __init__(self, config, lstm_embedding_size = 768, dense_hidden_size = 1536,
                    lstm_dropout_prob = 0.1, num_tags = 7):
        super(RoBertaSSC, self).__init__(config)
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base",  # We still use the 12-layer RoBERTA model to capture the semantics
            output_attentions=False,  # No need to returns attentions weights.
            output_hidden_states=False,  # No need to returns all hidden-states.
        )
        # self.dropout = torch.nn.Dropout(0.1)
        self.bilstm = torch.nn.LSTM(
            input_size=lstm_embedding_size,     # We use the BiLSTM layers to output the predictions
            hidden_size=dense_hidden_size // 2,     # specify the hidden_size for each BiLSTM layer
            batch_first=True,
            num_layers=2,   # specify the number of the BiLSTM layer
            dropout=lstm_dropout_prob,
            bidirectional=True  # make sure it is a bidirectional-LSTM layer
        )
        self.classifier = torch.nn.Linear(lstm_embedding_size * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):
        sequence_output, pooled_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask)

        bilstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(bilstm_output[:, -1, :])
        logits = logits.unsqueeze(1)

        outputs = (logits,)
        if labels is not None:
            labels = labels.unsqueeze(1)
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        # else: loss = 1
        # returns: (loss), scores
        return outputs

