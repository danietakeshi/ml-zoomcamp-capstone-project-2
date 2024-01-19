import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

input_file = 'ml-zoomcamp-qa-sentence-transformer'

print('Loading model...')
model = SentenceTransformer(f'./{input_file}')

print('Loading Embedding Database')
file_name = 'embeddings.parquet'
df = pd.read_parquet(f'./{file_name}')
candidate_answers = df.answer.tolist()
candidate_answers_embeddings = df.answer_embedding.tolist()
print('Embedding Database Loaded')

app = Flask('qa-query')

@app.route('/predict', methods=['POST'])
def predict():
    question = request.get_json()

    print('Generating embeddings')
    embeddings = model.encode([question['query']])
    similarity_scores = cosine_similarity([embeddings[0]], candidate_answers_embeddings)

    print('Creating probability dataframe')
    list_prob = []

    for answer, score in zip(
        candidate_answers,
        similarity_scores[0]
    ):
        list_prob.append([answer, score])
        
    df_prob = pd.DataFrame(list_prob, columns=['answer', 'score'])

    result = []
    for answer, score in df_prob.sort_values(by='score', ascending=False).head(3).values:
        result.append({'answer': answer, 'score': score})

    result = {'data': result}

    return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9696)