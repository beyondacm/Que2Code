# I Know What You Are Searching For: Code Snippet Recommendation from Stack Overflow Posts



Stack Overflow has been heavily used by software developers to seek programming-related information.
Typically, when developers encounter a technical problem, they formulate the problem as a query and use a search engine to obtain a list of possible relevant posts that may containuseful solutions to their problem. 
However, this kind of solution-seeking experience can be difficult and painful because the **_Query Mistmatch_** and **_Information Overload_** problems. To alleviate these challenges, in this work we present a query-driven code recommendation tool, named _Que2Code_, that identifies the best code snippets for a user query from Stack Overflow posts. 

Our model contains two stages: 

- _Semantically-Equivalent Question Retrieval_
- _Best Code Snippet Recommendation_


Our model has two sub-components, i.e., **QueryRewriter** and **CodeSelector**. **QueryRewriter** can qualitatively retrieve semantically-equivalent questions, and the **CodeSelector** can quantitatively rank the most relevant code snippets to the top of the recommendation candidates.


















