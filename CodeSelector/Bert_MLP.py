from utils import *

class Config(object):

    def __init__(self): 
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.num_classes = 2
        self.bert_path = './Model'
        self.hidden_size = 768
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.batch_size = 16
        self.num_epochs = 7 


class Model(nn.Module):
    
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True 
        self.fc0 = nn.Linear(2*config.hidden_size, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

    # def forward(self, input_ids, attention_mask, token_type_ids):
    def forward(self, qc0_pair, qc1_pair):
        
        qc0_input_ids, qc0_input_mask, qc0_input_types = qc0_pair[0], qc0_pair[1], qc0_pair[2] 
        qc1_input_ids, qc1_input_mask, qc1_input_types = qc1_pair[0], qc1_pair[1], qc1_pair[2] 
        '''
        qc0_last_hidden_states = self.bert(input_ids = qc0_input_ids, \
                                       attention_mask = qc0_input_mask, \
                                       token_type_ids = qc0_input_types) 

        qc1_last_hidden_states = self.bert(input_ids = qc1_input_ids, \
                                       attention_mask = qc1_input_mask, \
                                       token_type_ids = qc1_input_types) 
        qc0_features = qc0_last_hidden_states[0][:,0,:]
        qc1_features = qc1_last_hidden_states[0][:,0,:]
        # print('qc0_features:', type(qc0_features), qc0_features.shape)
        # print('qc1_features:', type(qc1_features), qc1_features.shape)
        features = torch.cat((qc0_features, qc1_features), dim=1)
        # print("features:", type(features), features.shape)
        '''
        _, qc0_pooled = self.bert(input_ids = qc0_input_ids, \
                                       attention_mask = qc0_input_mask, \
                                       token_type_ids = qc0_input_types) 

        _, qc1_pooled = self.bert(input_ids = qc1_input_ids, \
                                       attention_mask = qc1_input_mask, \
                                       token_type_ids = qc1_input_types) 

        features = torch.cat((qc0_pooled, qc1_pooled), dim=1)
        features = self.fc0(features)
        features = self.fc1(features)
        out = self.fc2(features)
        return out


