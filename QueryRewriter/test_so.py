import torch
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
# sentences = ["how to disable a tkinter canvas object", "how to check if a server is up or not in python?"]

# sentence = "how to disable a tkinter canvas object"
# sentence = "Binary file downloaded is unreadable"
# sentence = "Tensorflow GPU gettting stop while training the model"
# sentence = "How to click on a button with javascript using python or vba"
# sentence = "Read/Write Google Sheets in python from a Google Cloud"

# sentence = "How to split a web address"
sentence = "Binary file downloaded is unreadable"
# sentence = "How to click on a button with javascript using python or vba"
# sentence = "memory leak caused by large objects in python"

sentence = sentence.strip().lower()
print("sentence:", sentence)
text =  "paraphrase: " + sentence + " </s>"

max_len = 256
encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
print("encoding:", type(encoding))
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
print("input_ids:", type(input_ids), input_ids.shape)
print("attention_masks:", type(attention_masks), attention_masks.shape)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
beam_outputs = model.generate(\
	input_ids=input_ids, \
    attention_mask=attention_masks,\
	do_sample=True,\
	max_length=256,\
	top_k=100,\
	top_p=0.98,\
	early_stopping=True,\
	num_return_sequences=7\
)


print ("\nOriginal Question ::")
print (sentence)
print ("\n")
print ("Paraphrased Questions :: ")
final_outputs =[]

for beam_output in beam_outputs:
	sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
	if sent.lower() != sentence.lower() and sent not in final_outputs:
		final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
	print("{}: {}".format(i, final_output))

