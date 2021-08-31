import torch
import os


def predict(prediction_dataloader, test_df, model, device, tokenizer, args):
    # Prediction on test set

    #print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.argmax()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    id = test_df.id.values

    labeldict = {"Facts": 0, "Ruling by Lower Court": 1, "Argument": 2, "Statute": 3, "Precedent": 4,
                 "Ratio of the decision": 5, "Ruling by Present Court": 6}

    testlabel = []
    for i in range(0, 1905):
        testlabel.append(list(labeldict.keys())[list(labeldict.values()).index(predictions[i])])

    for i in range(0, 1905):
        file1 = open(args.save_path+'run_bert_bilstm_crf_epoch1_lr175_TESTONLY.txt', "a")
        file1.write(id[i])
        file1.write('\t')
        file1.write(testlabel[i])
        file1.write('\n')
        file1.close()

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = args.save_path +'model_save_berta_bilstm_crf_epoch1_lr175_TESTONLY'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


