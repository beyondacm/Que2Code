import pickle
import random
import numpy as np
import pandas as pd
from scipy import spatial
from T5_model import T5_model  

random.seed(779)

def hits_count(candidate_ranks, k):
  '''
    candidate_ranks:
    list of candidates' ranks; one rank per question;
    length is a number of questions
    rank is a number from q to len(candidates of the question)
    e.g. [2, 3] means that first candidate has the rank 2,
                           second candidate has the rank 3
    k: number of top-ranked elements (k in hits@k metric)
    result: return Hits@k value for current ranking 
  '''
  count = 0
  for rank in candidate_ranks:
    if rank <= k:
      count += 1
  return count/(len(candidate_ranks)+1e-8)
 
def dcg_score(candidate_ranks, k):
  '''
    candidate_ranks:
    list of candidates' ranks; one rank per question;
    length is a number of questions
    rank is a number from q to len(candidates of the question)
    e.g. [2, 3] means that first candidate has the rank 2,
                           second candidate has the rank 3
    k: number of top-ranked elements (k in hits@k metric)
    
    result: return DCG@k value for current ranking
  '''
  score = 0
  for rank in candidate_ranks:
    if rank <= k:
      score += 1/np.log2(1+rank)
  return score/(len(candidate_ranks)+1e-8) 

def rank_candidates(question_vec, candidate_answers, model):
    '''
        question: a string
        candidate_answers: a list of strings
        result: a list of pairs (initial position in the list, question)
    '''
    # question_vec = model.encode(question)
    # print("question_vec:", question_vec)
    candidate_scores = []
    for answer in candidate_answers:
        answer_vec = model.encode(answer)
        # print("answer_vec:", answer_vec)
        score = 1 - spatial.distance.cosine(question_vec, answer_vec) 
        # print("score:", score)
        candidate_scores.append( score )
    # print("candidate_scores:", candidate_scores) 
    tl = [(i, candidate_answers[i], candidate_scores[i]) for i in range(len(candidate_answers))]
    # print("tl:", tl)
    stl = sorted(tl, key=lambda x:x[2], reverse=True)
    # print("stl:", stl)
    result = [(t[0], t[1]) for t in stl]
    # print("result:", result)
    return result 

with open('../../BM25-IR/eval_data.pkl', 'rb') as handler:
    eval_data = pickle.load(handler)
print('eval_data:', type(eval_data), len(eval_data))

with open('./eval_data_embed/eval_data_embed_0.pkl', 'rb') as handler:
    eval_data_embed = pickle.load(handler)
print('eval_data_embed:', type(eval_data_embed), len(eval_data_embed))

eval_pairs = []
for i in range(len(eval_data)):
    eval_pairs.append((eval_data[i]['src_qtitle'], \
                       eval_data[i]['tgt_qtitle'], \
                       eval_data[i]['similar_q1_title'], \
                       eval_data[i]['similar_q2_title'], \
                       eval_data[i]['similar_q3_title'], \
                       eval_data[i]['similar_q4_title']))

print('eval_pairs:', type(eval_pairs), len(eval_pairs))

model = T5_model() 

model_ranking = []
for i, e in enumerate(eval_pairs):
    if i not in eval_data_embed:
        continue
    src_question = e[0].strip().lower()
    tgt_question = e[1].strip().lower()
    similar_q1 = e[2].strip().lower()
    similar_q2 = e[3].strip().lower()
    similar_q3 = e[4].strip().lower()
    similar_q4 = e[5].strip().lower()
    question = src_question
    candidate_answers = []
    candidate_answers.append(tgt_question) 
    candidate_answers.append(similar_q1)
    candidate_answers.append(similar_q2)
    candidate_answers.append(similar_q3)
    candidate_answers.append(similar_q4)
    # print("question:", question)
    # print("candidate_answers:", candidate_answers)

    question_vec = eval_data_embed[i]['question_embedding']
    ranks = rank_candidates(question_vec, candidate_answers, model)
    # print("ranks:", ranks)
    model_ranking.append( [r[0] for r in ranks].index(0) + 1 )
    # print("model_ranking:", model_ranking)
    # break

print( "len of model_ranking:", len(model_ranking) )

def evaluate( sample_ranking ): 
    eval_dcg_scores = [] 
    eval_hits_count = []
    for k in [1, 2 ,3, 4, 5]:
        eval_dcg_scores.append( dcg_score(sample_ranking, k) )
        eval_hits_count.append( hits_count(sample_ranking, k)) 
    return eval_dcg_scores, eval_hits_count

dcg_scores_result = []
hits_count_result = []

for i in range(10):
    sample_ranking = random.sample(model_ranking, 200)
    eval_dcg_scores, eval_hits_count  = evaluate( sample_ranking ) 
    # Append
    dcg_scores_result.append(eval_dcg_scores) 
    hits_count_result.append(eval_hits_count)

dcg_scores_df = pd.DataFrame(dcg_scores_result, columns=['dcg1', 'dcg2', 'dcg3', 'dcg4', 'dcg5'])
hits_count_df = pd.DataFrame(hits_count_result, columns=['hits1', 'hits2', 'hits3', 'hits4', 'hits5'])
# print(dcg_scores_df)
# print(hits_count_df)
print(hits_count_df.describe())
print(dcg_scores_df.describe())

# for k in [1, 2, 3, 4, 5]:
#     print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(model_ranking, k), \
#     k, hits_count(model_ranking, k)))

