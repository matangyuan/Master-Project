from transformers import BertModel, RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np
import sys
import torch


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Converts a time in seconds and returns a string hh:mm:ss
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(train_dataloader, model, device, args):
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=args.adam_epsilon  # args.adam_epsilon  - default is 1e-8.w
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
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            loss, logits = model(input_ids=b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            total_train_loss += loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # # ========================================
        # #               Validation
        # # ========================================
        # # After the completion of each training epoch, measure our performance on
        # # our validation set.
    '''
        print("")
        print("Running Validation...")

        t0 = time.time()

        # # Put the model in evaluation mode--the dropout layers behave differently
        # # during evaluation.
        model.eval()

        # # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # # Evaluate data for one epoch
        for batch in validation_dataloader:

        #     # Unpack this training batch from our dataloader. 
        #     #
        #     # As we unpack the batch, we'll also copy each tensor to the GPU using 
        #     # the `to` method.
        #     #
        #     # `batch` contains three pytorch tensors:
        #     #   [0]: input ids 
        #     #   [1]: attention masks
        #     #   [2]: labels 
           b_input_ids = batch[0].to(device)
           b_input_mask = batch[1].to(device)
           b_labels = batch[2].to(device)

        #     # Tell pytorch not to bother with constructing the compute graph during
        #     # the forward pass, since this is only needed for backprop (training).
           with torch.no_grad():        

        #         # Forward pass, calculate logit predictions.
        #         # token_type_ids is the same as the "segment ids", which 
        #         # differentiates sentence 1 and 2 in 2-sentence tasks.
        #         # The documentation for this `model` function is here: 
        #         # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
               # Get the "logits" output by the model. The "logits" are the output
               # values prior to applying an activation function like the softmax.
               (loss, logits) = model(b_input_ids, 
                                  token_type_ids=None, 
                                  attention_mask=b_input_mask,
                                  labels=b_labels)

           # Accumulate the validation loss.
           total_eval_loss += loss.item()

           # Move logits and labels to CPU
           logits = logits.detach().cpu().numpy()
           label_ids = b_labels.to('cpu').numpy()

           # Calculate the accuracy for this batch of test sentences, and
           # accumulate it over all batches.
           #print(loss)
           logits=logits[0]
           total_eval_accuracy += flat_accuracy(logits, label_ids)


        # # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # # Calculate the average loss over all of the batches
        avg_val_loss = total_eval_loss / len(validation_dataloader)

       # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # # Record all statistics from this epoch.
        training_stats.append(
           {
               'epoch': epoch_i + 1,
               'Training Loss': avg_train_loss,
               'Valid. Loss': avg_val_loss,
               'Valid. Accur.': avg_val_accuracy,
               'Training Time': training_time,
               'Validation Time': validation_time
           }
       )
    '''
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return model
