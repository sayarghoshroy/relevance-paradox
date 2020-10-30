print("Begin Execution", flush = True)

import numpy as np
import matplotlib.pyplot as plt
import re
import cupy
import json
import os
import time
import sys
import spacy
import spacy_sentence_bert
import nltk
from tqdm import tqdm

print("Imports Completed", flush = True)

filename = 'cleaned_document.json'

with open(filename, 'r+') as f:
  sent_scores = json.load(f)

# Suppressing Outputs

# Grounding Spacy's Transformer Models

# In case Spacy is not installed
# !pip install -U spacy

# Restart Runtime for this to take effect
# !pip install spacy-transformers
# !pip install spacy_sentence_bert
# !pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.0.4/en_roberta_large_nli_stsb_mean_tokens-0.0.4.tar.gz

# !python -m spacy download en

# !python -m spacy download en_trf_bertbaseuncased_lg
# !pip install -Iv botocore==1.17


# Use GPU for the Spacy Models
# Make Sure the machine has cuda
spacy.require_gpu()


# Required Downloads for NLTK
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

valid_POS = ['CD', 'FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 
             'PRP$', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'UNK']
be_verbs = ['be', 'is', 'was', 'were']

def get_content_ratio(sentence):
    tokenized_set = nltk.word_tokenize(sentence)
    tagged_map = nltk.pos_tag(tokenized_set)

    count_content = 0
    count_total = len(tokenized_set)

    for elem in tagged_map:
      checker = elem[1]
      if checker in valid_POS and checker != 'POS' and elem[0] not in be_verbs:
          count_content += 1
          
    ratio = count_content / count_total
    return ratio

# Loading Spacy's Transformer Model

print("Loading Model", flush = True)

nlp = spacy_sentence_bert.load_model('en_roberta_large_nli_stsb_mean_tokens')

print("Model Loaded", flush = True)

def clean(txt):
  txt = txt.replace("\n", " ").replace("\r", " ")
  punc_list = '!"#$&*+,-./;?@\^_~)('
  t = str.maketrans(dict.fromkeys(punc_list, " "))
  txt = txt.translate(t)
  t = str.maketrans(dict.fromkeys("'`", ""))
  txt = txt.translate(t)

  return txt

def regtok(txt):
  txt = clean(txt)
  regex = re.compile(r'(\d+|\s+|=|}}|\|)')
  tokens = [token for token in regex.split(txt)]
  return tokens

limit = len([key for key in sent_scores.keys()])
print("Number of Documents: " + str(limit), flush = True)

# Initialize Experiment Logs

lengths = []
scores = []
similarity_scores = []
content_ratios = []

pdf_keys = []
too_large = []

start_time = 0

def save(doc_count, suppress = 1):
  if suppress != 1:
      print("Saving Logs at Document Count = " + str(doc_count), flush = True)

  global lengths
  global scores
  global similarity_scores
  global content_ratios
  global pdf_keys
  global too_large
  global start_time

  time_elapsed = round((time.time() - start_time) / 3600, 2)

  stats = {'docs': doc_count, 'datapoints': len(scores), 'hours': time_elapsed}

  with open('stats.json', 'w+') as f:
      json.dump(stats, f)

  with open('lengths.json', 'w+') as f:
      json.dump(lengths, f)

  with open('content_ratios.json', 'w+') as f:
      json.dump(content_ratios, f)

  with open('scores.json', 'w+') as f:
      json.dump(scores, f)

  with open('similarity.json', 'w+') as f:
      json.dump(similarity_scores, f)

  with open('pdf_keys.json', 'w+') as f:
      json.dump(pdf_keys, f)

  with open('too_large.json', 'w+') as f:
      json.dump(too_large, f)

# Run test for a Specific Number of Documents
# limit = 25

interval = 1000

print("Limit: " + str(limit) + ", Saving Interval: " + str(interval), flush = True)

doc_index = 0
start_time = time.time()

for key in tqdm(sent_scores.keys(), total = limit, disable = True):
    if doc_index >= limit:
        break
    doc_index += 1

    if doc_index % interval == 0:
        save(doc_index, suppress = 0)

    if key.lower().endswith('.pdf'):
        pdf_keys.append(key)
        continue
    # ignore PDFs

    doc = sent_scores[key]['sent_labels']
    if len(doc) > 100:
        too_large.append(key)
        continue
    # ignore very large documents
    # with > 100 sentences
    
    sent_embs = []
    full_sequence = ""

    valid_doc = False
    
    for unit in doc:
      if len(unit) != 3:
          # Missing Parameters
          continue
      try:
          if unit[2] == 1:
              # checking if it is a grammatical sentence
              temp_score = float(unit[1])
              temp_length = int(len(regtok(unit[0])))
              semantic_vec = nlp(unit[0])

              if temp_score > -1 and temp_score < 5000 and temp_length < 1000:
                  valid_doc = True
                  scores.append(temp_score)
                  lengths.append(temp_length)
                  content_ratios.append(get_content_ratio(unit[0].strip()))
                  sent_embs.append(semantic_vec)
                  full_sequence = full_sequence + unit[0].strip() + " "

      except Exception as e:
         continue
    
    if valid_doc == False:
        continue
    
    # Correctness Check
    # print("DOC: " + str(full_sequence.strip()))

    doc_emb = nlp(full_sequence.strip())
    for emb in sent_embs:
        similarity_scores.append(np.ndarray.tolist(cupy.asnumpy(doc_emb.similarity(emb))))

assert len(lengths) == len(scores)
assert len(content_ratios) == len(scores)
assert len(similarity_scores) == len(scores)
print("Number of Data Points: " + str(len(lengths)))

# Study: Importance versus Length
fig = plt.figure(figsize = (12, 12))
p = fig.add_subplot(111)
p.set_title('Importance versus Length')
p.set_xlabel('Length')
p.set_ylabel('Importance Score')
p.scatter(lengths, scores, color = 'cyan', marker = 'o', alpha = 0.25)
plt.savefig('importance_vs_length.png')
# plt.show()

# Study: Importance versus Content Ratios
fig = plt.figure(figsize = (12, 12))
p = fig.add_subplot(111)
p.set_title('Importance versus Content Ratio')
p.set_xlabel('Content Ratio')
p.set_ylabel('Importance Score')
p.scatter(content_ratios, scores, color = 'green', marker = 'o', alpha = 0.25)
plt.savefig('importance_vs_content_1.4k.png')
# plt.show()

# Study: Importance versus Semantic Similarity
fig = plt.figure(figsize = (12, 12))
p = fig.add_subplot(111)
p.set_title('Importance versus Semantic Similarity')
p.set_xlabel('Semantic Similarity')
p.set_ylabel('Importance Score')
p.scatter(similarity_scores[0: min(len(scores), len(similarity_scores))], scores[0: min(len(scores), len(similarity_scores))], color = 'red', marker = 'o', alpha = 0.25)
plt.savefig('importance_vs_similarity_1.4k.png')
# plt.show()

print("Number of URLs with too large text: " + str(len(too_large)), flush = True)
print("Number of PDFs: " + str(len(pdf_keys)), flush = True)

# Saving Final Results
save(doc_index, suppress = 0)

"""---

### Tricky Cases

- Document Number 1444: PDF File with 3k+ Sentence Units

---
"""

# ^_^ Thank You
