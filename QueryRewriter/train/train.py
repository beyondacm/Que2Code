from utils import *
from T5FineTuner import T5FineTuner
from T5FineTuner import LoggingCallback
from ParaphraseDataset import ParaphraseDataset  

# set arguments
args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='./Paraphrse_Pretrained/',
    tokenizer_name_or_path='./Paraphrse_Pretrained/',
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

tokenizer = T5Tokenizer.from_pretrained('./Paraphrse_Pretrained/')
# dataset = ParaphraseDataset(tokenizer, 'data', 'dev', 256)  

train_path = "./data/so_train.csv"
val_path = "./data/so_val.csv"

data_train = pd.read_csv(train_path, sep="\t")#.astype(str)
# print(data_train.head())
data_val = pd.read_csv(val_path, sep="\t")
print(data_train.shape, data_val.shape)

if not os.path.exists('t5_paraphrase'):
    os.makedirs('t5_paraphrase')

args_dict.update({'data_dir': 'data', 'output_dir': 't5_paraphrase', 'num_train_epochs':10,'max_seq_length':256})
args = argparse.Namespace(**args_dict) 
print("args_dict:")
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(\
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    # early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

# def get_dataset(tokenizer, type_path, args):
# 	return ParaphraseDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)

print ("Initialize model")
model = T5FineTuner(args)

trainer = pl.Trainer(**train_params)

print (" Training model")
trainer.fit(model)
print ("training finished")

print ("Saving model")
model.model.save_pretrained('t5_paraphrase')

print ("Model saved")
