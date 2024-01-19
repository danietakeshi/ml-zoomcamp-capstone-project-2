import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def create_embeddings_db(file_name: str) -> pd.DataFrame:
    print('Loading dataset')
    qa_train = pd.read_csv(f"./{file_name}", sep=";")
    candidate_answers = qa_train['answer'].to_list()

    print('creating embeddings')
    list_embeddings = []
    for enum, answer in enumerate(candidate_answers):
        print(f"embedding question {enum}")
        answer_embedding = model.encode(str(answer))
        list_embeddings.append([answer, answer_embedding])

    print('creating embeddings db')
    embedings_db = pd.DataFrame(list_embeddings, columns=['answer', 'answer_embedding'])
    embedings_db.to_parquet('embeddings.parquet')

    return embedings_db

input_file = 'ml-zoomcamp-qa-sentence-transformer'

print('Loading model...')
model = SentenceTransformer(f'./{input_file}')

print('Loading Embedding Database')
file_name = 'embeddings.parquet'
df = pd.read_parquet(f'./{file_name}')
candidate_answers = df.answer.tolist()
candidate_answers_embeddings = df.answer_embedding.tolist()

question = 'Is it Possible to use AWS insted of GCP?'

print('Generating embeddings')
embeddings = model.encode([question])
similarity_scores = cosine_similarity([embeddings[0]], candidate_answers_embeddings)

print('Creating probability dataframe')
list_prob = []

for answer, score in zip(
    candidate_answers,
    similarity_scores[0]
):
    list_prob.append([answer, score])
    
df_prob = pd.DataFrame(list_prob, columns=['answer', 'score'])

print('Printing Top 3 answers')
print('---')
result = []
for answer, score in df_prob.sort_values(by='score', ascending=False).head(3).values:
    result.append({'answer': answer, 'score': score})
    print(score)
    print()
    print(answer)
    print('---')

result = {'data': result}

print(result)
