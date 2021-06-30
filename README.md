# I Know What You Are Searching For: Code Snippet Recommendation from Stack Overflow Posts



Stack Overflow has been heavily used by software developers to seek programming-related information.
Typically, when developers encounter a technical problem, they formulate the problem as a query and use a search engine to obtain a list of possible relevant posts that may containuseful solutions to their problem. 
However, this kind of solution-seeking experience can be difficult and painful because the **_Query Mistmatch_** and **_Information Overload_** problems. To alleviate these challenges, in this work we present a query-driven code recommendation tool, named _Que2Code_, that identifies the best code snippets for a user query from Stack Overflow posts. 

![Workflow of Que2Code](./figures/workflow.png)

Our model contains two stages: 

- _Semantically-Equivalent Question Retrieval_
- _Best Code Snippet Recommendation_


Our model has two sub-components, i.e., **QueryRewriter** and **CodeSelector**. **QueryRewriter** can qualitatively retrieve semantically-equivalent questions, and the **CodeSelector** can quantitatively rank the most relevant code snippets to the top of the recommendation candidates.

## QueryRewriter
The idea of **QueryRewriter** is to use a rewritten version of a query question to cover a variety of different forms of semantically equivalent expressions. 
In particular, we first collect the duplicate question pairs from Stack Overflow, because duplicate questions can be considered as semantically-equivalent questions of various user descriptions.
We then frame this problem as a sequence-to-sequence learning problem, which directly maps a technical question to its corresponding duplicate question. 
We train a text-to-text transformer, named **QueryRewriter**, by using the collected duplicate question pairs.


To train the **QueryRewriter**, please download our duplicate question dataset from the following link: [Dataset Download Link](https://drive.google.com/drive/folders/1-qlk1clhgy1Lzx4BIE5bW5fmEQsFSMjv?usp=sharing)

```shell
cd QueryRewriter/train/
python prepare_data.py
python train.py
```  

Or, we have released the pre-trained model as described in the paper. You can use the following command to download our pretrained model: [Pretrained Model Download Link](https://drive.google.com/drive/folders/1-E8pPL3ze7jHkR4_J6htAPk7iN94yInt?usp=sharing)

The **QueryRewriter** can be easily used with the following way:

```
query = "how to iterate the list in python"
# Initialize the model 
QR_model = QueryRewriter()
# Get the embedding of a query
query_vec = QR_model.encode(query)
# Get the paraphrase questions of a query
paraphrase_q = QR_model.paraphrase(query)
```   

## CodeSelector














