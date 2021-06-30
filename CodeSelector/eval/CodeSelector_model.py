from utils import *
from Bert_MLP import Model, Config

class CodeSelector(object):
    
    def __init__(self):
        self.config = Config()
        self.model = self.load_model() 
        self.tokenizer = self.load_tokenizer()
        pass

    def load_model(self): 
        PATH = './model_save/epoch7/model.ckpt'
        model = Model(self.config).to(self.config.device)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        print('Model Loaded!')
        return model

    def load_tokenizer(self): 
        tokenizer = AutoTokenizer.from_pretrained('./Model')
        print("Tokenizer Loaded!")
        return tokenizer
        pass
    
    def encode_qc(self, question, cs):  
        encoded_qc = self.tokenizer(question, cs, padding=True, truncation=True, max_length=128, return_tensors='pt')
        return encoded_qc
    
    def get_score(self, question, cs):
        # encode qc_0
        encoded_qc0 = self.encode_qc(question, cs)
        qc0_input_ids = encoded_qc0['input_ids'] 
        qc0_token_type_ids = encoded_qc0['token_type_ids']
        qc0_attention_masks = encoded_qc0['attention_mask']

        # encode qc_1 
        encoded_qc1 = self.encode_qc('', '')
        qc1_input_ids = encoded_qc1['input_ids'] 
        qc1_token_type_ids = encoded_qc1['token_type_ids']
        qc1_attention_masks = encoded_qc1['attention_mask']

        # to device 
        b_qc0_input_ids = qc0_input_ids.to(self.config.device)
        b_qc0_input_mask = qc0_attention_masks.to(self.config.device)
        b_qc0_input_types = qc0_token_type_ids.to(self.config.device)
        b_qc1_input_ids = qc1_input_ids.to(self.config.device)
        b_qc1_input_mask = qc1_attention_masks.to(self.config.device)
        b_qc1_input_types = qc1_token_type_ids.to(self.config.device)
        with torch.no_grad():
            qc0 = (b_qc0_input_ids, b_qc0_input_mask, b_qc0_input_types)
            qc1 = (b_qc1_input_ids, b_qc1_input_mask, b_qc1_input_types)
            outputs = self.model(qc0, qc1)

        score  = outputs.data.cpu().numpy()[0][1]
        return score
    

    def get_candidate_scores(self, question, candidate_answers):
        candidate_scores = []
        for cs in candidate_answers:
            score = self.get_score(question, cs)
            candidate_scores.append(score)
        return candidate_scores


def main():
    cs_model = CodeSelector()
    with open('../../BM25-IR/eval_data.pkl', 'rb') as handler:
        eval_data = pickle.load(handler) 

    for k, v in eval_data.items():
        question = v['qtitle']
        candidate_answers = []
        best_code = v['best_code']
        similar_code1 = v['similar_code1']
        similar_code2 = v['similar_code2']
        similar_code3 = v['similar_code3']
        similar_code4 = v['similar_code4']
        candidate_answers.append(best_code) 
        candidate_answers.append(similar_code1) 
        candidate_answers.append(similar_code2)
        candidate_answers.append(similar_code3) 
        candidate_answers.append(similar_code4)
        candidate_scores = cs_model.get_candidate_scores(question, candidate_answers)
        print(candidate_scores)
        break
    pass

if __name__ == '__main__':
    main()

