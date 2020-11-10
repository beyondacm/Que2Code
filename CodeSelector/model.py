from utils import *
from Bert_MLP import Model, Config

class Our_Model(object):
    
    def __init__(self):
        self.config = Config()
        self.model = self.load_model() 
        self.tokenizer = self.load_tokenizer()
        pass

    def load_model(self): 
        # PATH = './model_save/epoch1/model.ckpt'
        PATH = './model_save/step1-6000/model.ckpt'
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

    def get_score(self, question, cs0, cs1): 
        # encoded_qc0
        encoded_qc0 = self.encode_qc(question, cs0)
        qc0_input_ids = encoded_qc0['input_ids'] 
        qc0_token_type_ids = encoded_qc0['token_type_ids']
        qc0_attention_masks = encoded_qc0['attention_mask']

        # encoded_qc1
        encoded_qc1 = self.encode_qc(question, cs1)
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

        # return the score for class-1 
        # print(outputs.data)
        score  = outputs.data.cpu().numpy()[0][1]
        # print("outputs:", outputs.data.cpu().numpy()[0][1])
        return score
    
    def get_scores_map(self, question, candidate_answers):
        '''
            question: a string
            candidate_answers: a list of strings
            result: each score for a candidate 
        '''
        scores_map = {}
        for i in range(len(candidate_answers)):
            for j in range(len(candidate_answers)):
                key = str(i) + '-' + str(j)
                if i == j:
                    scores_map[key] = 0
                    continue
                
                score = self.get_score(question, candidate_answers[i], candidate_answers[j])
                scores_map[key] = score
        return scores_map
    
    def get_candidate_scores(self, scores_map):
        '''
        '''
        candidate_scores = [] 
        candidate0_score = scores_map['0-0']+scores_map['0-1']+scores_map['0-2']+scores_map['0-3']+scores_map['0-4']
        candidate1_score = scores_map['1-0']+scores_map['1-1']+scores_map['1-2']+scores_map['1-3']+scores_map['1-4']
        candidate2_score = scores_map['2-0']+scores_map['2-1']+scores_map['2-2']+scores_map['2-3']+scores_map['2-4']
        candidate3_score = scores_map['3-0']+scores_map['3-1']+scores_map['3-2']+scores_map['3-3']+scores_map['3-4']
        candidate4_score = scores_map['4-0']+scores_map['4-1']+scores_map['4-2']+scores_map['4-3']+scores_map['4-4']
        candidate_scores.append(candidate0_score)
        candidate_scores.append(candidate1_score)
        candidate_scores.append(candidate2_score)
        candidate_scores.append(candidate3_score)
        candidate_scores.append(candidate4_score)
        return candidate_scores


def main():
    model = Our_Model()
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
        scores_map = model.get_scores_map(question, candidate_answers)
        candidate_scores = model.get_candidate_scores(scores_map)
        print(scores_map)
        print(candidate_scores)
        break
    pass

if __name__ == '__main__':
    main()

