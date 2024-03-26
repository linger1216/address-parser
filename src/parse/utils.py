import json
import numpy as np
import yaml
import torch
import pickle
import torch.nn as nn


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


def load_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)


def load_yaml(path):
  with open(path, 'r', encoding='utf-8') as f:
    return yaml.safe_load(f)


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


def group_tokens(tokens, category=''):
  stack = []
  group = []
  for i, token in enumerate(tokens):
    if category and token not in [f'B-{category}', f'I-{category}', f'E-{category}', f'S-{category}']:
      continue

    if token.startswith(f'S-{category}'):
      group.append([(token, i)])
      continue

    if len(stack) == 0:
      stack.append((token, i))
      continue

    last_token, _ = stack[-1]

    if (
      last_token.startswith(f'B-{category}') and token.startswith(f'I-{category}') or
      last_token.startswith(f'B-{category}') and token.startswith(f'E-{category}') or
      last_token.startswith(f'I-{category}') and token.startswith(f'I-{category}') or
      last_token.startswith(f'I-{category}') and token.startswith(f'E-{category}')):
      stack.append((token, i))
    else:
      group.append(stack)
      stack = [(token, i)]

  if len(stack) > 0:
    group.append(stack)

  return sorted(group, key=lambda x: x[0][1])


def batch_group_tokens(batch_tokens, category=''):
  batch_group = []
  for tokens in batch_tokens:
    batch_group.append(group_tokens(tokens, category))
  return batch_group


def trans_group(group, raw):
  # 如果group代表空, 预测或真实的那个结果,应该就是空.
  if len(group) == 0:
    return "", ""
  label = group[0][0].strip().split('-')[-1]
  start = group[0][1]
  end = group[-1][1]
  return ''.join(raw[start:end + 1]), label
