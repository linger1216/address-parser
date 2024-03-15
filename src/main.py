import os
import json
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torchsummary import summary
from collections import defaultdict

# =================================================================
# hyper parameters
# =================================================================
# label_path = '../input/labels.txt'
label_path = '../input/linan/label.txt'
vocab_path = '../input/vocab.pk'
embedding_path = '../input/embedding.npy'

# train_path = '../input/train.json'
train_path = '../input/linan/train.json'
# eval_path = '../input/eval.json'
eval_path = '../input/linan/eval.json'
# test_path = '../input/test.json'
test_path = '../input/linan/eval.json'


epochs = 5
num_layers = 2
batch_size = 1024
sequence_length = 50
input_size = 300
hidden_size = 200
dropout_rate = 0.4

# 有个坑, 词向量找不到元素, 会用0填充...我们要处理下.
padding_value = 0



# =================================================================
# 名词解释:
# label 代表标注的类别
# =================================================================


# =================================================================
# load resources
# =================================================================

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


def read_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)


def load_data(json_datas, sequence_length):
  datas = []
  labels = []
  for d in json_datas:
    text = d.get('text')
    datas.append(text)
    label = [tokens[1] for tokens in d.get('tokens')]
    if label:
      labels.append(label)
    else:
      raise ValueError(f'error label: {label}')

  for p_, y_ in zip(datas, labels):
    assert len(p_) == len(y_)
  return datas, labels


# =================================================================
# convert resources
# =================================================================

"""
tensor of label
text -> id -> padding -> tensor
"""


def tensor_label(targets, label_map, padding_value, sequence_length):
  target_ids = []
  for target in targets:
    ids = []
    for label in target:
      id = label_map.get(label)
      if id:
        ids.append(label_map.get(label))
      else:
        raise ValueError(f'error label: {label}')
    if len(ids) <= sequence_length:
      ids.extend([padding_value] * (sequence_length - len(ids)))
    else:
      raise ValueError(f'error length: {len(ids)} {ids}')
    target_ids.append(ids)
  return torch.tensor(target_ids, dtype=torch.int64)


# 将target id转成one hot编码
# def one_hot_target_id(target_ids, label_num):
#   target_ids_tensor = torch.tensor(target_ids, dtype=torch.int64)
#   one_hot = F.one_hot(target_ids_tensor, num_classes=label_num)
#   return one_hot

# 将中文文本(训练/测试)转成词向量
# 在使用预训练的词向量进行嵌入后，张量的维度将发生变化。假设输入数据的形状为 (2, 50)，其中 2 是批量大小（batch size），50 是序列长度（sequence length）。
# 预训练的词向量维度为 (18109, 300)，其中 18109 是词汇表中的词语数量，300 是每个词语的嵌入维度。
# 在进行嵌入后，输出张量的形状将是 (2, 50, 300)。这是因为每个词语都被嵌入为一个长度为 300 的向量，而输入序列中有 50 个词语，所以输出张量的形状将是 (2, 50, 300)。
# 其中 2 是批量大小，50 是序列长度，300 是每个词语的嵌入维度。
"""
text -> tokens -> vocab -> padding -> tensor
"""


def tensor_data(vocabulary, datas, padding_value, sequence_length):
  tokens = []
  for text in datas:
    ids = [vocabulary.get(token, 0) for token in text]
    ids += [padding_value] * (sequence_length - len(ids))
    tokens.append(ids)
  return torch.tensor(tokens, dtype=torch.int64)


"""
input: text -> tokens -> vocab -> padding -> tensor
do: tensor -> embeddings
"""


def embedding_data(tensors, embeddings):
  return embeddings(tensors)


# =================================================================
# BI GRU + CRF model
# =================================================================
class BIGRU_CRF(nn.Module):
  def __init__(self, input_size, hidden_size, dropout_rate, labels_size):
    super(BIGRU_CRF, self).__init__()

    # input_size = embedding_dim(300) 是说每个词的维度是300, 作为输入x的维度
    # hidden_size = 200, 是指每个GRU单元的输出维度是200, 这个是人为指定的, 经验值
    # batch_first = True, 是指输入的数据的第一个维度是batch_size, 也就是说输入的数据的维度是(batch_size, sequence_length, embedding_dim)
    # bidirectional = True, 是指双向GRU
    # ----------------------------------------------------------------
    # Layer (type)               Output Shape           Param #
    # ================================================================
    # GRU-1  [[-1, 50, 400],     [-1, 2, 200]]               0
    # GRU-2  [[-1, 50, 400],     [-1, 2, 200]]               0
    # Dropout-3                  [-1, 50, 400]               0
    # Linear-4                   [-1, 50, 108]          43,308
    # ================================================================

    # 定义层
    self.gru_1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                        batch_first=True, bidirectional=True, dropout=dropout_rate)  # 双向GRU层
    self.gru_2 = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers,
                        batch_first=True, bidirectional=True, dropout=dropout_rate)  # 双向GRU层
    # self.dropout = nn.Dropout(dropout_rate)  # Dropout层
    self.fc = nn.Linear(hidden_size * 2, labels_size)

  def forward(self, x):
    # h0 = torch.zeros(num_layers * 2, x.size(0), hidden_size)

    out, h1 = self.gru_1(x)  # 第一层GRU
    out, _ = self.gru_2(out, h1)  # 第二层GRU
    # x = self.dropout(x)  # Dropout
    # out = out[:, -1, :]
    out = self.fc(out)  # 全连接层
    return out


# =================================================================
# EVAL Functions
# =================================================================

def group_tokens(tokens, category):
  stack = []
  group = []
  for i, token in enumerate(tokens):
    if token not in [f'B-{category}', f'I-{category}', f'E-{category}', f'S-{category}']:
      continue

    if token == f'S-{category}':
      group.append([(token, i)])
      continue

    if len(stack) == 0:
      stack.append((token, i))
      continue

    last_token, _ = stack[-1]

    if (
      last_token == f'B-{category}' and token == f'I-{category}' or
      last_token == f'B-{category}' and token == f'E-{category}' or
      last_token == f'I-{category}' and token == f'I-{category}' or
      last_token == f'I-{category}' and token == f'E-{category}'):
      stack.append((token, i))
    else:
      group.append(stack)
      stack = [(token, i)]

  if len(stack) > 0:
    group.append(stack)

  return sorted(group, key=lambda x: x[0][1])


def batch_group_tokens(batch_tokens, category):
  batch_group = []
  for tokens in batch_tokens:
    batch_group.append(group_tokens(tokens, category))
  return batch_group


"""
统计的结果
predict_group: 预测的group
y_group: 真实的group
note: 所有的group只保留了特定的category


correct, error_result, error_encode, error_unknown, precision, recall
"""


# 比较复杂
def statistics_group(predict_groups, y_groups):
  correct = []
  error_result = []
  error_encode = []
  error_unknown = []
  precision = 0
  recall = 0

  # 没预测出来也算一种错
  if len(predict_groups) == 0 and len(y_groups) > 0:
    for y_group in y_groups:
      error_result.append({'predict': [], 'y': y_group})
    return correct, error_result, error_encode, error_unknown, precision, recall

  # 预测错了, 真实分类是没有
  if len(predict_groups) > 0 and len(y_groups) == 0:
    error_result = []
    for predict_group in predict_groups:
      error_result.append({'predict': predict_group, 'y': []})
    return correct, error_result, error_encode, error_unknown, precision, recall

  if len(predict_groups) == 0 and len(y_groups) == 0:
    return correct, error_result, error_encode, error_unknown, precision, recall

  # 预测失败的
  for predict_one_group in predict_groups:
    # 预测成功的, 理想状态
    if predict_one_group in y_groups:
      correct.append(predict_one_group)
      continue

    # 上面的条件排除了预测成功的, 剩下只有预测失败的和编码错误的

    # 只有一个token
    if len(predict_one_group) == 1:
      # 是S开头的, 就不是编码错
      if predict_one_group[0][0].startswith('S'):
        error_result.append({'predict': predict_one_group, 'y': y_groups[0]})
      else:
        # 不是S开头的, 是编码错
        error_unknown.append(predict_one_group)
    else:
      """
      长度大于1的情况: 有几种可能:
      1. 预测错误: 如有交集, 也就是当前的group下标在y中有元素跟他下标有交集, 那么是预测错误
      2. 如果没有交集
      2.1 编码错误: 不能编码
      2.2 未知错误: 可以编码
      """
      intersection = False
      intersection_index = -1
      for i, y_one_group in enumerate(y_groups):
        if len(set([x[1] for x in predict_one_group]) & set([x[1] for x in y_one_group])) > 0:
          intersection = True
          intersection_index = i
          break

      if intersection:
        error_result.append({'predict': predict_one_group, 'y': y_groups[intersection_index]})
      else:
        # 判断是否可以编码
        if predict_one_group[0][0].startswith('B') and predict_one_group[-1][0].startswith('E'):
          error_unknown.append(predict_one_group)
        else:
          error_encode.append(predict_one_group)

  # 计算准确率
  precision = len(correct) / len(predict_groups)
  # 计算召回率
  recall = len(correct) / len(y_groups)
  return correct, error_result, error_encode, error_unknown, precision, recall


def batch_statistics_group(batch_predict_group, batch_y_group):
  corrects = []
  error_results = []
  error_encodes = []
  error_unknowns = []
  precisions = []
  recalls = []
  precision_mean = 0
  recall_mean = 0

  for predict_group, y_group in zip(batch_predict_group, batch_y_group):
    correct, error_result, error_encode, error_unknown, precision, recall = statistics_group(predict_group, y_group)
    corrects.append(correct)
    error_results.append(error_result)
    error_encodes.append(error_encode)
    error_unknowns.append(error_unknown)
    precisions.append(precision)
    recalls.append(recall)

  if len(precisions) > 0:
    precision_mean = sum(precisions) / len(precisions)

  if len(recalls) > 0:
    recall_mean = sum(recalls) / len(recalls)

  return corrects, error_results, error_encodes, error_unknowns, precisions, recalls, precision_mean, recall_mean


def trans_group(group, raw):
  # 如果group代表空, 预测或真实的那个结果,应该就是空.
  if len(group) == 0:
    return ""
  start = group[0][1]
  end = group[-1][1]
  return ''.join(raw[start:end + 1])

def remove_trailing_padding(a, padding_value):
  # 从末尾开始找到第一个不是padding_value的元素的位置
  index = next((i for i, value in enumerate(reversed(a)) if value != padding_value), len(a))

  # 返回截取后的数组
  return a[:-index] if index > 0 else a

def eval_decode(predict, y, r, vocab_map, label_map):
  """
  predict shape: (batch, actual_length)
  y shape: (batch, sequence_length)
  r shape: (batch, sequence_length)
  """
  inverse_labels_map = {v: k for k, v in label_map.items()}
  inverse_vocab_map = {v: k for k, v in vocab_map.items()}

  assert len(predict) == len(y) == len(r)
  y = y.numpy()
  r = r.numpy()

  size = len(predict)
  batch_predict_labels = []
  batch_y_labels = []
  batch_r_tokens = []


  for i in range(size):
    predict_items = predict[i]
    predict_labels = []
    predict_items = remove_trailing_padding(predict_items, padding_value)
    for item in predict_items:
      label = inverse_labels_map[item]
      predict_labels.append(label)
    batch_predict_labels.append(predict_labels)

    y_items = y[i]
    y_labels = []
    y_items = remove_trailing_padding(y_items, padding_value)
    for j in y_items:
      label = inverse_labels_map[j]
      y_labels.append(label)
    batch_y_labels.append(y_labels)

    r_items = r[i]
    r_tokens = []
    r_items = remove_trailing_padding(r_items, padding_value)
    for index, j in enumerate(r_items):
      token = inverse_vocab_map[j]
      r_tokens.append(token)
    batch_r_tokens.append(r_tokens)

  return batch_predict_labels, batch_y_labels, batch_r_tokens


def eval(eval_data_loader, model, padding_value, vocab_map, label_map):
  eval_corrects = defaultdict(list)
  eval_error_results = defaultdict(list)
  eval_error_encodes = defaultdict(list)
  eval_error_unknowns = defaultdict(list)
  eval_precisions = defaultdict(list)
  eval_recalls = defaultdict(list)

  for X, y, r in eval_data_loader:
    predict = model(X)
    mask = (y != padding_value)
    predict = crf.decode(predict, mask)
    predict, y, r = eval_decode(predict, y, r, vocab_map, label_map)

    corrects, error_results, error_encodes, error_unknowns, precisions, recalls = batch_eval_score(predict, y, r)
    for k, v in corrects.items():
      eval_corrects[k].extend(v)
    for k, v in error_results.items():
      eval_error_results[k].extend(v)
    for k, v in error_encodes.items():
      eval_error_encodes[k].extend(v)
    for k, v in error_unknowns.items():
      eval_error_unknowns[k].extend(v)
    for k, v in precisions.items():
      eval_precisions[k].extend(v)
    for k, v in recalls.items():
      eval_recalls[k].extend(v)

  assert len(eval_precisions.keys()) == len(eval_recalls.keys())
  score = {}
  for key in eval_precisions.keys():
    precision = sum(eval_precisions[key]) / len(eval_precisions[key])
    recall = sum(eval_recalls[key]) / len(eval_recalls[key])
    if precision == 0 and recall == 0:
      f1_score = 0
    else:
      f1_score = (2 * precision * recall) / (precision + recall)
    score[key] = (precision, recall, f1_score)

  return score, eval_corrects, eval_error_results, eval_error_encodes, eval_error_unknowns


"""
predict: batch_size, tokens (平铺的标签)
"""


def batch_eval_score(batch_predict, batch_y, batch_r):
  assert len(batch_predict) == len(batch_y) == len(batch_r)
  for p_, y_, r_ in zip(batch_predict, batch_y, batch_r):
    assert len(p_) == len(y_) == len(r_)

  # 根据y的内容来生成unique_category
  corrects = defaultdict(list)
  error_results = defaultdict(list)
  error_encodes = defaultdict(list)
  error_unknowns = defaultdict(list)
  precisions = defaultdict(list)
  recalls = defaultdict(list)
  for predict_tokens, y_tokens, r in zip(batch_predict, batch_y, batch_r):
    y_tokens_category = [token.strip().split('-')[-1] for token in y_tokens]
    unique_y_tokens_category = list(set(y_tokens_category))
    for category in unique_y_tokens_category:
      predict_group_tokens = group_tokens(predict_tokens, category)
      y_group_tokens = group_tokens(y_tokens, category)
      correct, error_result, error_encode, error_unknown, precision, recall = statistics_group(predict_group_tokens,
                                                                                               y_group_tokens)
      if len(correct) > 0:
        corrects[category].append((correct, r))
      if len(error_result) > 0:
        error_results[category].append((error_result, r))
      if len(error_encode) > 0:
        error_encodes[category].append((error_encode, r))
      if len(error_unknown) > 0:
        error_unknowns[category].append((error_unknown, r))

      precisions[category].append(precision)
      recalls[category].append(recall)

  return corrects, error_results, error_encodes, error_unknowns, precisions, recalls


# =================================================================
# Train Functions
# =================================================================

# 定义比较完善的训练函数
def train(train_data_loader, eval_data_loader, model, optimizer, num_epoch,
          save_step_interval, eval_step_interval, model_save_path, resume, eval_save_path='../eval'):
  start_epoch = 0
  start_step = 0

  # eval result
  scores = defaultdict(list)
  errors = {}
  if resume:
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    num_epoch = checkpoint.get('num_epoch', num_epoch)
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    print(f'resume training from epoch {start_epoch}/{num_epoch} step {start_step} by {resume}')

  for epoch in range(start_epoch, num_epoch):
    num_batches = len(train_data_loader)
    for batch_index, (X, y, _) in enumerate(train_data_loader):
      step = num_batches * epoch + batch_index + 1
      if step <= start_step:
        continue

      optimizer.zero_grad()
      predict = model(X)
      mask = (y != padding_value)
      loss = -1 * crf(predict, y, mask=mask, reduction='mean')
      loss.backward()
      optimizer.step()

      print(f'Epoch {epoch + 1}/{num_epoch} Step {step}/{len(train_data_loader)} Loss: {loss.item()}')

      # save model by epoch end
      if step % save_step_interval == 0:
        os.makedirs(model_save_path, exist_ok=True)
        save_file = os.path.join(model_save_path, f'model_step_{step}.pt')
        torch.save({
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'epoch': epoch,
          'num_epoch': num_epoch,
          'step': step
        }, save_file)

      if step % eval_step_interval == 0:
        model.eval()
        score, corrects, error_results, error_encodes, error_unknowns = eval(eval_data_loader, model, padding_value, vocab_map, label_map)
        for category, val in score.items():
          print(f'Epoch {epoch + 1}/{num_epoch} Step {step}/{len(train_data_loader)} label: {category} precision: {val[0]} recall: {val[1]} f1_score: {val[2]}')
          scores[category].append({
            'epoch': epoch + 1,
            'num_epoch': num_epoch,
            'step': step,
            'precision': val[0],
            'recall': val[1],
            'f1_score': val[2]
          })

        # 这里的替换其实没关系, 因为只有最后一次的结果会被保存
        # 也就是, 模型的最后一次输出结果, 才代表最后结果
        errors['corrects'] = corrects
        errors['error_results'] = error_results
        errors['error_encodes'] = error_encodes
        errors['error_unknowns'] = error_unknowns
        model.train()

  eval_save_result(scores, errors, eval_save_path)


def eval_save_result(scores, errors, eval_save_path):
  """
  save eval result
  1. save scores
  2. save datas
  """

  eval_path = os.path.join(eval_save_path, f'eval_{datetime.now().strftime("%Y%m%d%H%M%S")}')
  # save scores
  eval_scores_path = os.path.join(eval_path, 'scores')
  os.makedirs(eval_scores_path, exist_ok=True)
  for category, val in scores.items():
    with open(os.path.join(eval_scores_path, f'{category}_scores.csv'), 'w', encoding='utf-8') as f:
      f.write("epoch,num_epoch,step,precision,recall,f1_score\n")
      for v in val:
        f.write(f"{v['epoch']},{v['num_epoch']},{v['step']},{v['precision']},{v['recall']},{v['f1_score']}\n")
  # save datas
  eval_data_path = os.path.join(eval_path, 'datas')
  os.makedirs(eval_data_path, exist_ok=True)
  # write correct
  for category, datas in errors['corrects'].items():
    if len(datas) == 0:
      continue
    with open(os.path.join(eval_data_path, f'{category}_corrects.csv'), 'w', encoding='utf-8') as f:
      f.write(f"correct\n")
      for groups, raw in datas:
        for group in groups:
          text = trans_group(group, raw)
          f.write(f"{text}\n")

  # write error result
  for category, datas in errors['error_results'].items():
    if len(datas) == 0:
      continue
    with open(os.path.join(eval_data_path, f'{category}_error_results.csv'), 'w', encoding='utf-8') as f:
      f.write(f"predict, y\n")
      for groups, raw in datas:
        for group in groups:
          predict_text = trans_group(group['predict'], raw)
          y_text = trans_group(group['y'], raw)
          f.write(f"{predict_text},{y_text}\n")

  # write error encode
  for category, datas in errors['error_encodes'].items():
    if len(datas) == 0:
      continue
    with open(os.path.join(eval_data_path, f'{category}_error_encodes.csv'), 'w', encoding='utf-8') as f:
      f.write(f"error_encode\n")
      for groups, raw in datas:
        for group in groups:
          text = trans_group(group, raw)
          f.write(f"{text}\n")

  # write error unknown
  for category, datas in errors['error_unknowns'].items():
    if len(datas) == 0:
      continue
    with open(os.path.join(eval_data_path, f'{category}_error_unknowns.csv'), 'w', encoding='utf-8') as f:
      f.write(f"error_unknown\n")
      for groups, raw in datas:
        for group in groups:
          text = trans_group(group, raw)
          f.write(f"{text}\n")


if __name__ == '__main__':
  logo = """
   __    ____  ____  ____  ____  ___  ___    ____  ____    __    ____  _  _ 
  /__\  (  _ \(  _ \(  _ \( ___)/ __)/ __)  (_  _)(  _ \  /__\  (_  _)( \( )
 /(__)\  )(_) ))(_) ))   / )__) \__ \\__ \    )(   )   / /(__)\  _)(_  )  ( 
(__)(__)(____/(____/(_)\_)(____)(___/(___/   (__) (_)\_)(__)(__)(____)(_)\_)
  """
  print(logo)

  _, label_map, _ = load_labels(label_path)
  vocab_map = load_vocabs(vocab_path)
  embeddings = load_embeddings(embedding_path)

  """
  data definition
  """
  train_data, train_label = load_data(read_json(train_path), sequence_length)
  train_data_tensor = tensor_data(vocab_map, train_data, padding_value, sequence_length)
  train_data_embedding = embedding_data(train_data_tensor, embeddings)
  train_label = tensor_label(train_label, label_map, padding_value, sequence_length)
  train_dataset = torch.utils.data.TensorDataset(train_data_embedding, train_label, train_data_tensor)
  train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  eval_data, eval_label = load_data(read_json(eval_path), sequence_length)
  eval_data_tensor = tensor_data(vocab_map, eval_data, padding_value, sequence_length)
  eval_data_embedding = embedding_data(eval_data_tensor, embeddings)
  eval_label = tensor_label(eval_label, label_map, padding_value, sequence_length)
  eval_dataset = torch.utils.data.TensorDataset(eval_data_embedding, eval_label, eval_data_tensor)
  eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

  test_data, test_label = load_data(read_json(test_path), sequence_length)
  test_data_tensor = tensor_data(vocab_map, test_data, padding_value, sequence_length)
  test_data_embedding = embedding_data(test_data_tensor, embeddings)
  test_label = tensor_label(test_label, label_map, padding_value, sequence_length)
  test_dataset = torch.utils.data.TensorDataset(test_data_embedding, test_label, test_data_tensor)
  test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  """
  model definition
  """

  crf = CRF(num_tags=len(label_map), batch_first=True)
  model = BIGRU_CRF(input_size, hidden_size, dropout_rate, len(label_map))
  summary(model, (sequence_length, input_size), batch_size=batch_size)
  optimizer = torch.optim.Adam(model.parameters())

  train(train_data_loader, eval_data_loader, model, optimizer, epochs, save_step_interval=10, eval_step_interval=5,
        model_save_path='../models', resume='', eval_save_path='../eval')

  # 读档
  # train(train_data_loader, eval_data_loader, model, optimizer, epochs, save_step_interval=10, eval_step_interval=10,
  #       model_save_path='../models', resume='../models/model_step_430.pt', eval_save_path='../eval')

  print("all done.")