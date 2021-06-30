from utils import *
from Bert_MLP import Model, Config
# from Bert_CNN import Model, Config

def save_model(epoch, model, training_stats):
    # Saving & Loading Fine-tuned Model
    ## Saving best-practices: if you use defaults names for the model, 
    ## you can reload it using from_pretrained()

    base_dir = './model_save/epoch' + str(epoch) + '/'
    # sub_dir = 'epoch' + str(epoch) +'/model.ckpt' 
    output_dir = base_dir + 'model.ckpt' 
    ## Create output directory if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(base_dir + "training_stats.json")
    # df_stats.to_pickle(output_dir + "training_stats.pkl")
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

def save_model_step(step, model, training_stats):
    
    base_dir = './model_save/step' + str(step) + '/'  
    output_dir = base_dir + 'model.ckpt' 
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Saving model to %s" % output_dir)
    torch.save(model.state_dict(), output_dir)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(base_dir + "training_stats.json")


# Load the iterator  
with open('./Data/train_dataloader.pkl', 'rb') as handle:
    train_dataloader = pickle.load(handle)

with open('./Data/validation_dataloader.pkl', 'rb') as handle:
    validation_dataloader = pickle.load(handle)

print("dataloader loaded!")


config = Config()
model = Model(config).to(config.device)
print("Model created!")

# Optimizer & Learning Rate Scheduler 
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the training data.
epochs = config.num_epochs 
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, \
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# We are ready to kick off the training
# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

print("Training start ...")
# For each epoch
for epoch_i in range(0, epochs): 
    # ==========================
    #       Training 
    # ==========================

    # Perform one full pass over the training set.
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


        # Progress update every 40 batches
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # break

        # Save by step size    
        if step % 1000 == 0 and not step == 0:
            # Record all statistics from this epoch.
            training_stats.append(\
            {'epoch': epoch_i + 1, \
             'step': step, \
             # 'Training Loss': avg_train_loss, \
             # 'Training Time': training_time, \
            })
            step_marker = str(epoch_i +1 ) + '-' + str(step)
            save_model_step(step_marker, model, training_stats)
            # break
        
        if step % 10000 == 0 and not step == 0:
            break

        # Unpack this training batch from our dataloader.  
        # 
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 'to' method
        # 
        # `batch` contains seven pytorch tensors:
        #   [0]: qc0 input ids 
        #   [1]: qc0 attention masks
        #   [2]: qc0 token type ids
        #   [3]: qc1 input ids 
        #   [4]: qc1 attention masks
        #   [5]: qc1 token type ids
        #   [6]: labels 
        b_qc0_input_ids   = batch[0].to(config.device)
        b_qc0_input_mask  = batch[1].to(config.device)
        b_qc0_input_types = batch[2].to(config.device)
        b_qc1_input_ids   = batch[3].to(config.device)
        b_qc1_input_mask  = batch[4].to(config.device)
        b_qc1_input_types = batch[5].to(config.device)
        b_labels = batch[6].to(config.device)
        # print('batch qc0 input_ids:', type(b_qc0_input_ids), b_qc0_input_ids.shape)
        # print('batch qc0 input_mask:', type(b_qc0_input_mask), b_qc0_input_mask.shape)
        # print('batch qc0 input_types:', type(b_qc0_input_types), b_qc0_input_types.shape)
        # print('batch qc1 input_ids:', type(b_qc1_input_ids), b_qc1_input_ids.shape)
        # print('batch qc1 input_mask:', type(b_qc1_input_mask), b_qc1_input_mask.shape)
        # print('batch qc1 input_types:', type(b_qc1_input_types), b_qc1_input_types.shape)
        # print('batch labels:', type(b_labels), b_labels.shape)
        
        model.zero_grad()

        b_qc0 = (b_qc0_input_ids, b_qc0_input_mask, b_qc0_input_types)
        b_qc1 = (b_qc1_input_ids, b_qc1_input_mask, b_qc1_input_types)
        b_outputs = model(b_qc0, b_qc1)
        # print('batch outputs:', type(b_outputs), b_outputs.shape)
        # exit()

        loss = F.cross_entropy(b_outputs, b_labels)
        # print('loss:', type(loss), loss, loss.item())

        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        # break
    # exit()
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    
    # After the completion of each training epoch, measure our performance on
    # our validation set. 
    
    print("")
    print("Running Validation...")
    
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.  
        # 
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 'to' method
        # 
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_qc0_input_ids   = batch[0].to(config.device)
        b_qc0_input_mask  = batch[1].to(config.device)
        b_qc0_input_types = batch[2].to(config.device)
        b_qc1_input_ids   = batch[3].to(config.device)
        b_qc1_input_mask  = batch[4].to(config.device)
        b_qc1_input_types = batch[5].to(config.device)
        b_labels = batch[6].to(config.device)


        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # values prior to applying an activation function like the softmax.
            b_qc0 = (b_qc0_input_ids, b_qc0_input_mask, b_qc0_input_types)
            b_qc1 = (b_qc1_input_ids, b_qc1_input_mask, b_qc1_input_types)
            b_outputs = model(b_qc0, b_qc1)
            # b_outputs = model(b_input_ids, b_input_mask, b_input_types)
            # print("b_outputs:", type(b_outputs), b_outputs.shape)

        loss = F.cross_entropy(b_outputs, b_labels)
             
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # move labels to CPU 
        preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
        # print("preds:", type(preds), preds.shape)
        labels = b_labels.to('cpu').numpy()
        # print("labels:", type(labels), labels.shape)
        # print(preds)
        # print(labels)

        # Calculate the accuracy for this batch of test sentences, and
        total_eval_accuracy += flat_accuracy(preds, labels)
        # break

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    
    # Record all statistics from this epoch.
    training_stats.append(\
        {'epoch': epoch_i + 1, \
         'Training Loss': avg_train_loss, \
         'Valid. Loss': avg_val_loss, \
         'Valid. Accur.': avg_val_accuracy, \
         'Training Time': training_time, \
         'Validation Time': validation_time
        })
    
    save_model(epoch_i + 1, model, training_stats)
    # exit()
    # break

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

