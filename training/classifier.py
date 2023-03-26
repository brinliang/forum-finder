import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pickle


# read data
nforums = 148
df = pd.read_csv('stackexchange.csv', usecols=['Forum', 'Title'], nrows=nforums*1000)
df['Title'] = df['Title'].apply(lambda x: str(x))

# create labels
label2index = {}
index2label = {}
for i in range(nforums):
  label2index[df['Forum'][i * 1000]] = i
  index2label[i] = df['Forum'][i * 1000]

# create bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# define dataloader
class Dataset(torch.utils.data.Dataset):
  def __init__(self, df):
    self.labels = [label2index[label] for label in df['Forum']]
    self.texts = [tokenizer(text, padding='max_length', max_length = 70, truncation=True, return_tensors="pt") for text in df['Title']]

  def classes(self):
    return self.labels

  def __len__(self):
    return len(self.labels)

  def get_batch_labels(self, idx):
    return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    batch_texts = self.get_batch_texts(idx)
    batch_y = self.get_batch_labels(idx)
    return batch_texts, batch_y


# define classifier
class BertClassifier(nn.Module):
  def __init__(self, dropout=0.5):
    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, nforums)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, input_id, mask):
    _, x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    x = self.dropout(x)
    x = self.linear(x)
    x = self.relu(x)
    x = self.softmax(x)
    return x

  def inference(self, input_id, mask, top_k):
      x = self.forward(input_id, mask)
      x = torch.topk(x, k=top_k, dim=1)
      return x


# define training loop
def train(model, train_data, val_data, learning_rate, batch_size, epochs):
  # set up dataloader
  train, val = Dataset(train_data), Dataset(val_data)
  train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

  # define loss function
  criterion = nn.CrossEntropyLoss()
  
  # define optimizer
  optimizer = Adam(model.parameters(), lr=learning_rate)

  # use gpu if available
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  # track progress
  train_losses = []
  train_accuracy = []
  val_losses = []
  val_accuracy = []


  # training loop
  for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label in tqdm(train_dataloader):
      train_label = train_label.to(device)
      mask = train_input['attention_mask'].to(device)
      input_id = train_input['input_ids'].squeeze(1).to(device)
      output = model(input_id, mask)
      batch_loss = criterion(output, train_label.long())
      total_loss_train += batch_loss.item()
      acc = (output.argmax(dim=1) == train_label).sum().item()
      total_acc_train += acc
      model.zero_grad()
      batch_loss.backward()
      optimizer.step()

    total_acc_val = 0
    total_loss_val = 0

    with torch.no_grad():
      for val_input, val_label in val_dataloader:
        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)
        output = model(input_id, mask)
        batch_loss = criterion(output, val_label.long())
        total_loss_val += batch_loss.item()
        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc

    print(f'Epochs: {epoch_num + 1} \
            | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
    
    train_losses.append(total_loss_train / len(train_data))
    train_accuracy.append(total_acc_train / len(train_data))
    val_losses.append(total_loss_val / len(val_data))
    val_accuracy.append(total_acc_val / len(val_data))
  
  return train_losses, train_accuracy, val_losses, val_accuracy


def evaluate(model, test_data):
  # set up dataloader
  test = Dataset(test_data)
  test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

  # use gpu if available
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    model = model.cuda()

  # test
  total_acc_test = 0
  with torch.no_grad():
    for test_input, test_label in test_dataloader:
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)
      output = model(input_id, mask)
      acc = (output.argmax(dim=1) == test_label).sum().item()
      total_acc_test += acc

  print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
  
  return total_acc_test / len(test_data)


# train val test split
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=0), [int(.8*len(df)), int(.9*len(df))])

# create model
model = BertClassifier()

# training hyperparameters
EPOCHS = 1
LR = 1e-6
BATCH_SIZE = 2

# run training loop
train_losses, train_accuracies, val_losses, val_accuracies = train(model, df_train, df_val, LR, BATCH_SIZE, EPOCHS)
test_accuracy = evaluate(model, df_test)


# save logs
logs = {
  'epochs': EPOCHS,
  'train_losses': train_losses,
  'train_accuracies': train_accuracies,
  'val_losses': val_losses,
  'val_accuracies': val_accuracies,
  'test_accuracy': test_accuracy
}

with open('logs.pickle', 'wb') as f:
  pickle.dump(logs, f)

# save model
torch.save(model.state_dict(), 'model.pt')
