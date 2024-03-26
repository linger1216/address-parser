import bentoml
import yaml
from torchcrf import CRF

from utils import load_labels, load_vocabs, load_embeddings, tensor_data, embedding_data, remove_trailing_padding, \
  batch_group_tokens, trans_group

if __name__ == '__main__':
  with open('config.yaml', 'r', encoding='utf-8') as f:
    config = f.read()

  d = yaml.load(config, Loader=yaml.FullLoader)

  label_path = d['input']['label_path']
  vocab_path = d['input']['vocab_path']
  embedding_path = d['input']['embedding_path']
  sequence_length = d['params']['sequence_length']

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