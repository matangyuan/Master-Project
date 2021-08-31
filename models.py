from transformers import BertConfig, RobertaConfig, XLNetConfig
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from transformers import BertModel,RobertaForSequenceClassification, RobertaModel,RobertaForSequenceClassification, AdamW, RobertaConfig
from torchcrf import CRF


def mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    # Tokenize all of the sentences and map the tokens to their word IDs.
    test_input_ids = []
    test_attention_masks = []

    # For every sentence...
    for sent in test_sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        test_input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        test_attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    test_input_ids = torch.cat(test_input_ids, dim=0)
    test_attention_masks = torch.cat(test_attention_masks, dim=0)
    test_labels = torch.tensor(test_labels)

    # Set the batch size.
    test_batch_size = 1

    # Create the DataLoader.
    prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=test_batch_size)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(1 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning RoBERTA on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = args.batch_size

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, prediction_dataloader


def tokenize_BERT_sentence(sentences, labels, test_sentences, test_labels, args):
    configuration = BertConfig()
    # Load the bert-base-cased tokenizer.
    print('Loading bert-base-cased tokenizer:')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    # Print the original sentence.
    print(' Original Sentence: ', sentences[0])
    # Print the sentence split into tokens.
    print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
    # Print the sentence mapped to token ids.
    print('Token IDs of the Sentence: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Maximal sentence length: ', max_len)

    train_dataloader, prediction_dataloader = mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args)

    return train_dataloader, prediction_dataloader, tokenizer


def tokenize_RoBERTa_sentence(sentences, labels, test_sentences, test_labels, args):
    configuration = RobertaConfig()
    # Load the RoBERTa tokenizer.
    print('Loading RoBERTa tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    # Print the original sentence.
    print(' Original Sentence: ', sentences[0])
    # Print the sentence split into tokens.
    print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
    # Print the sentence mapped to token ids.
    print('Token IDs of the Sentence: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Maximal sentence length: ', max_len)

    train_dataloader, prediction_dataloader = mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args)

    return train_dataloader, prediction_dataloader, tokenizer

def tokenize_XLNet_sentence(sentences, labels, test_sentences, test_labels, args):
    configuration = XLNetConfig()
    # Load the XLNetTokenizer tokenizer.
    print('Loading XLNetTokenizer tokenizer...')
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    # Print the original sentence.
    print(' Original Sentence: ', sentences[0])
    # Print the sentence split into tokens.
    print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
    # Print the sentence mapped to token ids.
    print('Token IDs of the Sentence: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Maximal sentence length: ', max_len)

    train_dataloader, prediction_dataloader = mask_sentence(sentences, labels, test_sentences, test_labels, tokenizer, args)

    return train_dataloader, prediction_dataloader, tokenizer


class BertSSC(BertModel):
    def __init__(self, config, lstm_embedding_size = 768, dense_hidden_size = 1536,
                    lstm_dropout_prob = 0.1, num_tags = 7):
        super(BertSSC, self).__init__(config)
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",  # Use the bert-base-uncased model
            output_attentions=False,  # Determine whether the model returns attentions weights.
            output_hidden_states=False,  # Determine whether the model returns all hidden-states.
        )
        # self.dropout = torch.nn.Dropout(0.1)
        self.bilstm = torch.nn.LSTM(
            input_size=lstm_embedding_size,
            hidden_size=dense_hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=lstm_dropout_prob,
            bidirectional=True
        )
        self.classifier = torch.nn.Linear(lstm_embedding_size * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
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
            "roberta-base",  # Use the 12-layer RoBERTA model, with an uncased vocab.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        # self.dropout = torch.nn.Dropout(0.1)
        self.bilstm = torch.nn.LSTM(
            input_size=lstm_embedding_size,
            hidden_size=dense_hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=lstm_dropout_prob,
            bidirectional=True
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

