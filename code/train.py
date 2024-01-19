import os
import pandas as pd
import numpy as np
import torch
import warnings
import pickle

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

qa_train = pd.read_csv("./capstone-2-qa-dataset.csv", sep=";")

candidate_answers = qa_train['answer'].to_list()

model_id = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model = SentenceTransformer(model_id)

train_examples = []

for course, year, question, answer in qa_train.values:
    train_examples.append(InputExample(texts=[str(question), str(answer)]))
    
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.MultipleNegativesRankingLoss(model=model)

num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps)

output_file = 'ml-zoomcamp-qa-sentence-transformer'

print(f'Saving the model on {output_file}')

model.save(output_file)

question = 'Is it Possible to use AWS insted of GCP?'

list_aswers = candidate_answers
embeddings = model.encode([question] + candidate_answers)
similarity_scores = cosine_similarity([embeddings[0]], embeddings[1:])
print(question)
print()
print(candidate_answers[np.argmax(similarity_scores)])