import pickle
import numpy as np
import pandas as pd
import random
from model import Our_Model  
from scipy import spatial

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

def rank_candidates(candidate_answers, candidate_scores):
    '''
        question: a string
        candidate_answers: a list of strings
        result: a list of pairs (initial position in the list, question)
    '''
    # question_vec = model.encode(question)
    # print("question_vec:", question_vec)
    # candidate_scores = []
    # for answer in candidate_answers:
        # answer_vec = model.encode(answer)
        # print("answer_vec:", answer_vec)
        # score = 1 - spatial.distance.cosine(question_vec, answer_vec) 
        # print("score:", score)
        # candidate_scores.append( score )
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

eval_pairs = []
for k, v in eval_data.items():
    question = v['qtitle']
    best_code = v['best_code']
    similar_code1 = v['similar_code1']
    similar_code2 = v['similar_code2']
    similar_code3 = v['similar_code3']
    similar_code4 = v['similar_code4']
  
    eval_pairs.append(( question, \
                        best_code, \
                        similar_code1, \
                        similar_code2, \
                        similar_code3, \
                        similar_code4))
print('eval_pairs:', type(eval_pairs), len(eval_pairs))


def get_model_ranking(model, sampled_pairs):
    model_ranking = []
    for i, e in enumerate(sampled_pairs):
        question = e[0]
        best_code = e[1]
        similar_code1 = e[2]
        similar_code2 = e[3]
        similar_code3 = e[4]
        similar_code4 = e[5]
        candidate_answers = []
        candidate_answers.append(best_code) 
        candidate_answers.append(similar_code1)
        candidate_answers.append(similar_code2)
        candidate_answers.append(similar_code3)
        candidate_answers.append(similar_code4)

        scores_map = model.get_scores_map(question, candidate_answers)
        candidate_scores = model.get_candidate_scores(scores_map) 
        # print("question:", question)
        # print("candidate_answers:", candidate_answers)
        ranks = rank_candidates(candidate_answers, candidate_scores)
        # print("ranks:", ranks)
        model_ranking.append( [r[0] for r in ranks].index(0) + 1 )
        # print("model_ranking:", model_ranking)
        # break
    return model_ranking

def evaluate( model_ranking ):
    eval_dcg_scores = []
    eval_hits_count = []
    for k in [1, 2, 3, 4, 5]:
        eval_dcg_scores.append( dcg_score(model_ranking, k) )
        eval_hits_count.append( hits_count(model_ranking, k)) 
    return eval_dcg_scores, eval_hits_count


model = Our_Model() 

model_ranking = get_model_ranking(model, eval_pairs)
# print(type(model_ranking), len(model_ranking))
# print(model_ranking)
with open('./model_ranking.pkl', 'wb') as handler:
    pickle.dump(model_ranking, handler)

'''
dcg_scores_result = []
hits_count_result = []

for i in range(10):
    sampled_pairs = random.sample(eval_pairs, 200)
    model_ranking = get_model_ranking(model, sampled_pairs)
    eval_dcg_scores, eval_hits_count= evaluate( model_ranking )
    dcg_scores_result.append( eval_dcg_scores )
    hits_count_result.append( eval_hits_count )
    break

dcg_scores_df = pd.DataFrame(dcg_scores_result, columns=['dcg1', 'dcg2', 'dcg3', 'dcg4', 'dcg5'])
hits_count_df = pd.DataFrame(hits_count_result, columns=['hits1', 'hits2', 'hits3', 'hits4', 'hits5'])
print(hits_count_df.describe())
print(dcg_scores_df.describe())
'''
# for k in [1, 2, 3, 4, 5]:
#     print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(model_ranking, k), \
#                                               k, hits_count(model_ranking, k)))
