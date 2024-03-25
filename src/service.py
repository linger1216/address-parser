import os
import bentoml
from torchcrf import CRF
from bentoml.io import JSON
from utils import (load_yaml, load_labels, load_vocabs, load_embeddings,
                       tensor_data, embedding_data, remove_trailing_padding,
                       batch_group_tokens, trans_group)

def fix_path(path):
  if not os.path.exists(path):
    fixed_path = os.path.join('./src', path)
    return fixed_path
  return path

config_path = fix_path('config.yaml')

d = load_yaml(config_path)

label_path = fix_path(d['input']['label_path'])
vocab_path = fix_path(d['input']['vocab_path'])
embedding_path = fix_path(d['input']['embedding_path'])
sequence_length = d['params']['sequence_length']

_, label_map, _ = load_labels(label_path)
vocab_map = load_vocabs(vocab_path)
inverse_labels_map = {v: k for k, v in label_map.items()}
inverse_vocab_map = {v: k for k, v in vocab_map.items()}
embeddings = load_embeddings(embedding_path)
crf = CRF(num_tags=len(label_map), batch_first=True)

parse_runner = bentoml.pytorch.get("parse:latest").to_runner()
svc = bentoml.Service('parse', runners=[parse_runner])

print('init done.')

@svc.api(input=JSON(), output=JSON())
def parse(datas):
  tensors = tensor_data(vocabulary=vocab_map, datas=datas, padding_value=0, sequence_length=sequence_length)
  inputs = embedding_data(tensors, embeddings)
  result = parse_runner.run(inputs)
  predict = crf.decode(result, tensors != 0)

  batch_predict_labels = []
  for i, text in enumerate(predict):
    predict_labels = []
    predict_items = remove_trailing_padding(text, 0)
    for item in predict_items:
      label = inverse_labels_map[item]
      predict_labels.append(label)
    batch_predict_labels.append(predict_labels)

  batch_groups = batch_group_tokens(batch_predict_labels)

  batch_predict = []
  for i, groups in enumerate(batch_groups):
    predict = []
    for group in groups:
      text, label = trans_group(group, datas[i])
      predict.append({'text': text, 'label': label})
    batch_predict.append(predict)

  return batch_predict
