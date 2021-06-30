from utils import *
from Bert_MLP import Config

# Import DataSet
df = pd.read_csv('./Data/train_label_data', \
                 delimiter='\t', \
                 header = None, \
                 names = ['qid', 'question', 'aid0', 'cs0', 'aid1', 'cs1', 'label'])
print(type(df), df.shape)
print(df['label'].value_counts())

# Get the lists of questions 
questions = df.question.values.tolist()
cs0 = df.cs0.values.tolist()
cs1 = df.cs1.values.tolist()
labels = df.label.values.tolist()

print('questions:', type(questions), len(questions))
print('cs0:', type(cs0), len(cs0))
print('cs1:', type(cs1), len(cs1))
print('labels:', type(labels), len(labels))

# Tokenize & Input Formatting
## Import model/tokenizer 
## Load the BERT model
print("Loading BERT Model...")
bert_model = BertModel.from_pretrained('./Model')
bert_model.cuda()
print("Loading BERT Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('./Model')
# tokenizer = tokenizer_class.from_pretrained('./Model', do_lower_case=True)


# Required Formatting
## 1. sentences to ids
## 2. Padding & Truncating                          
## 3. Attention Masks
## 4. 
# Combine question + cs0 as the first inputs
encoded_qc0 = tokenizer(questions, cs0, padding=True, truncation=True, max_length=128, return_tensors='pt')
print("encoded_qc0:", type(encoded_qc0), len(encoded_qc0))
qc0_input_ids = encoded_qc0['input_ids']
qc0_token_type_ids = encoded_qc0['token_type_ids']
qc0_attention_masks = encoded_qc0['attention_mask']

print("qc0_input_ids:", type(qc0_input_ids), qc0_input_ids.shape)
print("qc0_type_ids:", type(qc0_token_type_ids), qc0_token_type_ids.shape)
print("qc0_attn_mask:", type(qc0_attention_masks), qc0_attention_masks.shape)

# Convert list to numpy array
qc0_input_ids = qc0_input_ids.cpu().detach().numpy()
qc0_token_type_ids = qc0_token_type_ids.cpu().detach().numpy()
qc0_attention_masks = qc0_attention_masks.cpu().detach().numpy()
print("qc0_input_ids:", type(qc0_input_ids), qc0_input_ids.shape )
print("qc0_type_ids:", type(qc0_token_type_ids), qc0_token_type_ids.shape )
print("qc0_attn_mask:", type(qc0_attention_masks), qc0_attention_masks.shape )


encoded_qc1 = tokenizer(questions, cs1, padding=True, truncation=True, max_length=128, return_tensors='pt')

qc1_input_ids = encoded_qc1['input_ids']
qc1_token_type_ids = encoded_qc1['token_type_ids']
qc1_attention_masks = encoded_qc1['attention_mask']

# Convert list to numpy array
qc1_input_ids = qc1_input_ids.cpu().detach().numpy()
qc1_token_type_ids = qc1_token_type_ids.cpu().detach().numpy()
qc1_attention_masks = qc1_attention_masks.cpu().detach().numpy()
print("qc1_input_ids:", type(qc1_input_ids), qc1_input_ids.shape )
print("qc1_type_ids:", type(qc1_token_type_ids), qc1_token_type_ids.shape )
print("qc1_attn_mask:", type(qc1_attention_masks), qc1_attention_masks.shape )

labels = np.asarray(labels)

with open('./Data/encoded_qc0.pkl', 'wb') as handle: 
    pickle.dump(encoded_qc0, handle) 

with open('./Data/encoded_qc1.pkl', 'wb') as handle:
    pickle.dump(encoded_qc1, handle)

with open('./Data/labels.pkl', 'wb') as handle:
    pickle.dump(labels, handle)

# Training and Validation Split on qc0
# Use 97% for training and 3% for validation
train_qc0_inputs, validation_qc0_inputs, train_labels, validation_labels = train_test_split(qc0_input_ids, \
                                                                                    labels, \
                                                                                    random_state=2018, \
                                                                                    test_size=0.03)
# Do the same for attention_mask
train_qc0_masks, validation_qc0_masks, _, _ = train_test_split(qc0_attention_masks, \
                                                       labels, \
                                                       random_state=2018, \
                                                       test_size = 0.03)

# Do the same for token_type_ids
train_qc0_types, validation_qc0_types, _, _ = train_test_split(qc0_token_type_ids, \
                                                       labels, \
                                                       random_state=2018, \
                                                       test_size = 0.03)

# Training and Validation Split on qc1
# Use 97% for training and 3% for validation
train_qc1_inputs, validation_qc1_inputs, _, _ = train_test_split(qc1_input_ids, \
                                                                 labels, \
                                                                 random_state=2018, \
                                                                 test_size=0.03)
# Do the same for attention_mask
train_qc1_masks, validation_qc1_masks, _, _ = train_test_split(qc1_attention_masks, \
                                                       labels, \
                                                       random_state=2018, \
                                                       test_size = 0.03)

# Do the same for token_type_ids
train_qc1_types, validation_qc1_types, _, _ = train_test_split(qc1_token_type_ids, \
                                                       labels, \
                                                       random_state=2018, \
                                                       test_size = 0.03)

# Convert to Pytorch Data Types
train_qc0_inputs = torch.tensor(train_qc0_inputs)
train_qc0_masks = torch.tensor(train_qc0_masks)
train_qc0_types = torch.tensor(train_qc0_types)

train_qc1_inputs = torch.tensor(train_qc1_inputs)
train_qc1_masks = torch.tensor(train_qc1_masks)
train_qc1_types = torch.tensor(train_qc1_types)

validation_qc0_inputs = torch.tensor(validation_qc0_inputs)
validation_qc0_masks = torch.tensor(validation_qc0_masks)
validation_qc0_types = torch.tensor(validation_qc0_types)

validation_qc1_inputs = torch.tensor(validation_qc1_inputs)
validation_qc1_masks = torch.tensor(validation_qc1_masks)
validation_qc1_types = torch.tensor(validation_qc1_types)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

print(type(train_qc0_inputs), train_qc0_inputs.shape)
print(type(train_qc0_masks), train_qc0_masks.shape)
print(type(train_qc0_types), train_qc0_types.shape)

print(type(train_qc1_inputs), train_qc1_inputs.shape)
print(type(train_qc1_masks), train_qc1_masks.shape)
print(type(train_qc1_types), train_qc1_types.shape)

print(type(train_labels), train_labels.shape)


# We'll also create an iterator for our dataset using the torch DataLoader class.
# This helps save on memory during training 
# unlike for loop, with an iterator the entire dataset does not need to be loaded into memory

config = Config()
# batch_size = 32
batch_size = config.batch_size
print("batch_size:", batch_size)

# Create the DataLoader for our training set.
train_data = TensorDataset(train_qc0_inputs, train_qc0_masks, train_qc0_types, \
                           train_qc1_inputs, train_qc1_masks, train_qc1_types, \
                           train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
print(type(train_dataloader))

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_qc0_inputs, validation_qc0_masks, validation_qc0_types, \
                                validation_qc1_inputs, validation_qc1_masks, validation_qc1_types, \
                                validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
print(type(validation_dataloader))

# Save DataLoader
with open('./Data/train_dataloader.pkl', 'wb') as handle:
    pickle.dump(train_dataloader, handle)

with open('./Data/validation_dataloader.pkl', 'wb') as handle:
    pickle.dump(validation_dataloader, handle)

print("Finished!")
