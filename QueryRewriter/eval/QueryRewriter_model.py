import torch
import pickle
import numpy as np
from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import T5Tokenizer, T5Model
# from transformers import T5Tokenizer, T5EncoderModel

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class QueryRewriter(object):
    
    def __init__(self):
        self.device = self.get_device()
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.input_embeddings = self.load_input_embeddings()
        print( type(self.input_embeddings) )
        pass

    def get_device(self): 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def load_model(self):
        model = T5ForConditionalGeneration.from_pretrained('./t5_paraphrase')
        # model = T5Model.from_pretrained('./t5_paraphrase')
        model = model.to(self.device)
        return model 

    def load_tokenizer(self):
        # tokenizer = T5Tokenizer.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('./t5_paraphrase')
        # tokenizer = T5Tokenizer.from_pretrained('./Paraphrse_Pretrained')
        return tokenizer

    def load_input_embeddings(self): 
        embeddings = self.model.get_input_embeddings()
        return embeddings 
   
    def encode(self, query):
        query = query.strip()
        # text =  "paraphrase: " + query + " </s>"
        text = query
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        ################ Alternative Method ############################
        # print("input_ids:", input_ids)
        outputs = self.input_embeddings(input_ids)
        # print("outputs:", outputs.shape)
        outputs = torch.squeeze(outputs)
        output_vec = outputs.cpu().detach().numpy()
        # print("output_vec:", output_vec.shape)
        output_vec = np.mean(output_vec, axis=0)
        # print("outputs:", type(outputs), len(outputs))
        return output_vec
    
    def paraphrase(self, query):
        query = query.strip()
        text =  "paraphrase: " + query + " </s>"
        max_len = 256
        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        beam_outputs = self.model.generate(\
                                    input_ids=input_ids, \
                                    attention_mask=attention_masks,\
                                    do_sample=True,\
                                    max_length=256, \
                                    top_k=120,\
                                    top_p=0.95,\
                                    early_stopping=True,\
                                    num_return_sequences=3\
                                    )
        final_outputs =[] 
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != query.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        return final_outputs
        

def main():
    query = "how to iterate the list in python"
    qr_model = QueryRewriter()
    query_vec = qr_model.encode(query)
    print("query_vec:", type(query_vec), query_vec.shape)
    
    paraphrase_q = qr_model.paraphrase(query)
    print(paraphrase_q)

    pass

if __name__ == "__main__":
    main()
    

