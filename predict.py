import torch
import os


def predict(prediction_dataloader, test_df, model, device, tokenizer, args):
    # Predicting the testing dataset

    #print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))

    # In evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # load the data to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from the dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # No need to calculate gradient
        with torch.no_grad():
            # Forward propagation to obtain prediction results
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.argmax()

        # Move the results to CPU
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
        file1 = open(args.save_path+args.model_name+'.txt', "a")
        file1.write(id[i])
        file1.write('\t')
        file1.write(testlabel[i])
        file1.write('\n')
        file1.close()

    # The path where the model is stored
    output_dir = args.save_path + args.model_name

    # Create if the directory does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Use `save_pretrained()` to save the trained model, model configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


