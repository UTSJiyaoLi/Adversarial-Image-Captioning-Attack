from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from coco import *
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rouge(summary, pred):
  # Get percision score
  scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
  scores_1, scores_2 = 0, 0
  for ref in summary:
    score_1 = scorer.score(ref, pred)['rouge1'][0]
    score_2 = scorer.score(ref, pred)['rouge2'][0]
    scores_1 += score_1
    scores_2 += score_2
  scores_1 = scores_1/len(summary)
  scores_2 = scores_2/len(summary)
  return scores_1, scores_2

def bleu1(reference, predict):
  reference = [ref.split() for ref in reference]
  predict = predict.split()
  bleu1_score = sentence_bleu(reference, predict, weights=(1, 0, 0, 0))
  return bleu1_score
  
def bleu2(reference, predict):
  reference = [ref.split() for ref in reference]
  predict = predict.split()
  bleu2_score = sentence_bleu(reference, predict, weights=(0.5, 0.5, 0, 0))
  return bleu2_score

def bleu4(reference, predict):
  reference = [ref.split() for ref in reference]
  predict = predict.split()
  # print(reference)
  score = sentence_bleu(reference, predict, weights=(0.25,0.25,0.25,0.25))
  # score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
  return score

def word_embeddings(sent_1, sent_2):
  """
  Input two sentences, and return a cosine score between them.
  """

  # Load pre-trained BERT model and tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

  # Tokenize and encode sentences
  tokens1 = tokenizer.encode_plus(sent_1, add_special_tokens=True, return_tensors='pt')
  tokens2 = tokenizer.encode_plus(sent_2, add_special_tokens=True, return_tensors='pt')

  # Obtain BERT embeddings
  with torch.no_grad():
      outputs1 = model(**tokens1)
      outputs2 = model(**tokens2)

  hidden_states1 = outputs1.hidden_states
  hidden_states2 = outputs2.hidden_states

  # Extract the final BERT embeddings (CLS token)
  sentence_embedding1 = hidden_states1[-1][:, 0, :]
  sentence_embedding2 = hidden_states2[-1][:, 0, :]

  # Calculate cosine similarity
  similarity = cosine_similarity(sentence_embedding1, sentence_embedding2)[0][0]

  # print("Cosine Similarity:", similarity)
  return similarity

def USE():
    pass

def bleurouge(bleu, rouge):
  if bleu + rouge == 0:
    return 0
  return (bleu * rouge)/(bleu + rouge)

def std_dev(data):
  # load data in list
  std_dev = np.std(data)
  return std_dev

if __name__ == '__main__':
  
  data = [0.23, 0.24, 0.23, 0.25, 0.19]
  print(std_dev(data))
