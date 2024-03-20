import numpy as np
import bentoml
import yaml
import torch
import pickle
import torch.nn as nn
from torchcrf import CRF


def load_labels(path):
  with open(path, "r", encoding="utf-8") as file:
    lines = file.readlines()
  labels = [line.strip() for line in lines]
  label_id_map = {label: i for i, label in enumerate(labels)}
  unique_category = [line.strip().split('-')[-1] for line in lines]
  return labels, label_id_map, list(set(unique_category))


def load_vocabs(path):
  with open(path, 'rb') as f:
    vocab = pickle.load(f)
  return vocab


def load_embeddings(path):
  embeddings = torch.tensor(np.load(path), dtype=torch.float32)
  embedding = nn.Embedding.from_pretrained(embeddings)
  return embedding


def tensor_data(vocabulary, datas, padding_value, sequence_length):
  tokens = []
  for text in datas:
    ids = [vocabulary.get(token, 0) for token in text]
    ids += [padding_value] * (sequence_length - len(ids))
    tokens.append(ids)
  return torch.tensor(tokens, dtype=torch.int64)

def embedding_data(tensors, embeddings):
  return embeddings(tensors)


def remove_trailing_padding(a, padding_value):
  # 从末尾开始找到第一个不是padding_value的元素的位置
  index = next((i for i, value in enumerate(reversed(a)) if value != padding_value), len(a))
  # 返回截取后的数组
  return a[:-index] if index > 0 else a

def group_tokens(tokens):
  stack = []
  group = []
  for i, token in enumerate(tokens):
    # start with S
    if token.startswith('S-'):
      group.append([(token, i)])
      continue

    if len(stack) == 0:
      stack.append((token, i))
      continue

    last_token, _ = stack[-1]

    if (
      last_token.startswith('B-') and token.startswith('I-') or
      last_token.startswith('B-') and token.startswith('E-') or
      last_token.startswith('I-') and token.startswith('I-') or
      last_token.startswith('I-') and token.startswith('E-')):
      stack.append((token, i))
    else:
      group.append(stack)
      stack = [(token, i)]

  if len(stack) > 0:
    group.append(stack)

  return sorted(group, key=lambda x: x[0][1])


def batch_group_tokens(batch_tokens):
  batch_group = []
  for tokens in batch_tokens:
    batch_group.append(group_tokens(tokens))
  return batch_group



def trans_group(group, raw):
  # 如果group代表空, 预测或真实的那个结果,应该就是空.
  if len(group) == 0:
    return ""
  label = group[0][0].strip().split('-')[-1]
  start = group[0][1]
  end = group[-1][1]
  return ''.join(raw[start:end + 1]), label



if __name__ == '__main__':
  with open('config.yaml', 'r', encoding='utf-8') as f:
    config = f.read()

  d = yaml.load(config, Loader=yaml.FullLoader)

  label_path = d['input']['label_path']
  vocab_path = d['input']['vocab_path']
  embedding_path = d['input']['embedding_path']
  num_epochs = d['params']['num_epochs']
  num_layers = d['params']['num_layers']
  batch_size = d['params']['batch_size']
  sequence_length = d['params']['sequence_length']
  input_size = d['params']['input_size']
  hidden_size = d['params']['hidden_size']
  dropout_rate = d['params']['dropout_rate']

  _, label_map, _ = load_labels(label_path)
  vocab_map = load_vocabs(vocab_path)
  inverse_labels_map = {v: k for k, v in label_map.items()}
  inverse_vocab_map = {v: k for k, v in vocab_map.items()}
  embeddings = load_embeddings(embedding_path)


  logo = """
   __    ____  ____  ____  ____  ___  ___    ____  ____    __    ____  _  _ 
  /__\  (  _ \(  _ \(  _ \( ___)/ __)/ __)  (_  _)(  _ \  /__\  (_  _)( \( )
  /(__)\  )(_) ))(_) ))   / )__) \__ \\__ \    )(   )   / /(__)\  _)(_  )  ( 
  (__)(__)(____/(____/(_)\_)(____)(___/(___/   (__) (_)\_)(__)(__)(____)(_)\_)
  """
  print(logo)

  crf = CRF(num_tags=len(label_map), batch_first=True)
  parse_runner = bentoml.pytorch.get("parse:latest").to_runner()
  parse_runner.init_local()

  datas = ['上海市静安区长临路187号', '上海市浦东新区玉兰馨园35号301']
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
      predict.append((text, label))
    batch_predict.append(predict)

  print(f'batch_predict={batch_predict}')
  print("all done")

# svc = bentoml.Service('parse', runners=[parse_runner])
# @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# def classify(input_series: np.ndarray) -> np.ndarray:
#   # 定义预处理逻辑
#   result = parse_runner.run(input_series)
#   # 定义后处理逻辑
#   return result
