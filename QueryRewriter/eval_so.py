import torch
import pickle
from transformers import T5ForConditionalGeneration,T5Tokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('./t5_paraphrase')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

# sentence = "Which course should I take to get started in data science?"
# sentence = "how to 'zoom' in on a section of the mandelbrot set?"
# sentence = "how to extract the following using pandas or regex"
# sentence = "how to check if an object is a list or tuple (but not string)?"
# sentence = "how to send a list of commands using 'os.system('command')' to cmd in python"
# sentence = "how to check if a server is up or not in python?"
# sentence = "how to bind engine to table with sqlalchemy"
# sentence = "how to disable a tkinter canvas object"

eval_data = {}
with open('/home/zhipeng/Code_Search/Question-Code/python/qid-question.txt', 'r') as f:
    for i, line in enumerate(f):
        qid, question = line.lower().strip().split('\t')
        key = i
        value = {}
        value['qid'] = qid
        value['question'] = question
        eval_data[key] = value

# sentences = []
# with open('/home/zhipeng/Code_Search/Duplicate-Question/python/data/eval-question', 'r') as fin:
#     for line in fin:
#         sentences.append(line.strip())
#for sentence in sentences:
paraphrase_result = {}
for i in range(len(eval_data)):
    key = i
    value = {}
    sentence = eval_data[key]['question'].strip()
    text =  "paraphrase: " + sentence + " </s>"
    max_len = 256
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(\
        input_ids=input_ids, \
        attention_mask=attention_masks,\
        do_sample=True,\
        max_length=256,\
        top_k=120,\
        top_p=0.98,\
        early_stopping=True,\
        num_return_sequences=5\
    )

    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    value['qid'] = eval_data[key]['qid']
    value['question'] = eval_data[key]['question'] 
    value['paraphrase'] = final_outputs
    paraphrase_result[key] = value
    print(i)
    # print(final_outputs)

# print(paraphrase_result)

with open('./paraphrase_result.pkl', 'wb') as handle:
    pickle.dump(paraphrase_result, handle)


