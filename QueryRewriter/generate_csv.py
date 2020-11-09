import pandas as pd

question1 = []
question2 = []

with open('/home/zhipeng/Code_Search/Duplicate-Question/python/data/src-train.txt', 'r') as src:
    for line in src:
        question1.append(line.strip())

with open('/home/zhipeng/Code_Search/Duplicate-Question/python/data/tgt-train.txt', 'r') as tgt:
    for line in tgt:
        question2.append(line.strip())

assert len(question1) == len(question2)
train_df = pd.DataFrame({'question1':question1, 'question2':question2})
train_df.to_csv('./paraphrase_data/so_train.csv', index=False)

question1 = []
question2 = []

with open('/home/zhipeng/Code_Search/Duplicate-Question/python/data/src-val.txt', 'r') as src:
    for line in src:
        question1.append(line.strip())

with open('/home/zhipeng/Code_Search/Duplicate-Question/python/data/tgt-val.txt', 'r') as tgt:
    for line in tgt:
        question2.append(line.strip())

assert len(question1) == len(question2)
val_df = pd.DataFrame({'question1':question1, 'question2':question2})
val_df.to_csv('./paraphrase_data/so_val.csv', index=False)

