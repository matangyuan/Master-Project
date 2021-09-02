from transformers import BertModel, RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np
import sys
import torch


def flat_accuracy(preds, labels):
    # This function returns the accuracy with given predictions and its labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Converts a time in seconds into hours:minutes:seconds
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format: hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(train_dataloader, model, device, args):
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,  # specify the learning rate
                      eps=args.adam_epsilon
                      )

    # Number of training epochs. As we use pre-trained models, dont need to run too many epoches
    # for training data.
    epochs = args.epochs
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    seed_val = 42   # Set the random seed value to ensure that the output is constant
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []   # Store statistical indicators such as training and evaluation loss, accuracy, etc.,
    total_t0 = time.time()

    # For every epoch...
    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('========Training========')

        # Count the training time of a single epoch
        t0 = time.time()

        # Reset the total training loss for each epoch
        total_train_loss = 0

        # Set the model to training mode. Not using the training interface
        # The performance of dropout and batchnorm layers in training and testing modes is different
        model.train()

        # For each batch
        for step, batch in enumerate(train_dataloader):

            # Every 40 iterations, output progress information
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Prepare input data and copy it to gpu
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Everytime before calculating the gradient, set the gradient to 0,
            # because the gradient of pytorch is cumulative
            model.zero_grad()
            # Forward propagation
            # This function will return different values according to different parameters.
            loss, logits = model(input_ids=b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Cumulative loss
            total_train_loss += loss
            # Backpropagation
            loss.backward()
            # Gradient clipping to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters
            optimizer.step()
            # Update learning rate
            scheduler.step()

        # Average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Training time of a single epoch
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # # ========================================
        # #               Validation
        # # ========================================
        # # Validation process the same as the training part

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return model
