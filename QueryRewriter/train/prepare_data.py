import pandas as pd
import pickle

train_question1 = []
train_question2 = []

# Make train data
with open('./duplicate_questions.train', 'r') as fin:
    for line in fin:
        src_qid, src_qtitle, tgt_qid, tgt_qtitle = line.strip().split('\t')
        duplicate_question = src_qtitle.lower()
        master_question = tgt_qtitle.lower()
        train_question1.append(duplicate_question)
        train_question2.append(master_question)
    # break

assert len(train_question1) == len(train_question2)
train_df = pd.DataFrame({'question1':train_question1, 'question2':train_question2})
train_df.to_csv('./so_train.csv', index=False, sep="\t") 


val_question1 = []
val_question2 = []

# Make test data
with open('./duplicate_questions.val', 'r') as fin:
    for line in fin:
        src_qid, src_qtitle, tgt_qid, tgt_qtitle = line.strip().split('\t')
        duplicate_question = src_qtitle.lower()
        master_question = tgt_qtitle.lower()
        val_question1.append(duplicate_question)
        val_question2.append(master_question)
        # break

assert len(val_question1) == len(val_question2)
val_df = pd.DataFrame({'question1':val_question1, 'question2':val_question2})
val_df.to_csv('./so_val.csv', index=False, sep="\t") 


