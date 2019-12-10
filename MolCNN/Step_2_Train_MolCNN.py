'''
Step 2: Train MolCNN

Input: 
    cnndata.p

Output:
    print estimated AUC results = 0.77 (we did not tune parameters)
'''

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')#处理警告
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.autograd as autograd
import time
from sklearn.metrics import roc_auc_score


datadict = pickle.load(open("cnndata.p", "rb" ))
train, y, test, vocabs, embed_size, weight, idx_to_word, word_to_idx = datadict["train"], datadict["y"], datadict["test"], datadict["vocabs"], datadict["embed_size"], datadict["weight"], datadict["idx_to_word"], datadict["word_to_idx"]
X = np.array(train.astype(int))

class TextCNN(nn.Module):

    def __init__(self, max_sent_len, embedding_dim, filter_sizes, num_filters, vocab_size, num_classes):


        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.word_embeddings = nn.Embedding.from_pretrained(weight)
        self.word_embeddings.weight.requires_grad = True

        conv_blocks = []
        for filter_size in filter_sizes:
            maxpool_kernel_size = max_sent_len - filter_size + 1
            conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=filter_size)
            component = nn.Sequential(
                conv1,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size))

            if use_cuda:
                component = component.cuda()
            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.word_embeddings(x)
        x = x.transpose(1, 2)   
        x_list = [conv_block(x) for conv_block in self.conv_blocks]

        out = torch.cat(x_list, 2)      
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        return F.softmax(self.fc(out), dim=1)

def evaluate_auc(model, xt, yt):
    with torch.no_grad():
        inputs = autograd.Variable(xt)
        preds = model(inputs)
        preds = preds[:,1].data
        yt = yt[:,1]
        if use_cuda:
            preds = preds.cuda()
        yt = yt.cpu().data
        preds = preds.cpu().data
        eval_acc = roc_auc_score(yt, preds)
        return eval_acc

use_cuda = torch.cuda.is_available()
embedding_dim = 300
num_filters = 100
filter_sizes = [3, 4, 5]
batch_size = 64
num_epochs = 10

np.random.seed(0)
torch.manual_seed(0)

vocab_size = len(word_to_idx)
max_sent_len = X.shape[1]
num_classes = 2 #y.shape[1]

print('vocab size       = {}'.format(vocab_size))
print('max sentence len = {}'.format(max_sent_len))
print('num of classes   = {}'.format(num_classes))

tic = time.time()
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print('train size = {}    test size = {}'.format(len(x_train), len(x_val)))
print('train ratio = {:.3f}    test ratio = {:.3f}'.format( sum(y_train)/len(y_train ), sum(y_val)/len(y_val)))

y_train = np.array([[0, 1] if label==1 else [1, 0] for label in y_train])
y_val = np.array([[0, 1] if label==1 else [1, 0] for label in y_val])

# numpy array to torch tensor
x_train = torch.from_numpy(x_train).long()
y_train = torch.from_numpy(y_train).float()
dataset_train = data_utils.TensorDataset(x_train, y_train)
train_loader = data_utils.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

x_val = torch.from_numpy(x_val).long()
y_val = torch.from_numpy(y_val).float()
if use_cuda:
    x_val = x_val.cuda()
    y_val = y_val.cuda()
    x_train = x_train.cuda()
    y_train = y_train.cuda()

cnnmodel = TextCNN(max_sent_len=max_sent_len,
                embedding_dim=embedding_dim,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                vocab_size=vocab_size,
                num_classes=num_classes)

if use_cuda:
    cnnmodel = cnnmodel.cuda()
optimizer = optim.Adam(cnnmodel.parameters(), lr=0.0002)

loss_fn = nn.BCELoss()

for epoch in range(num_epochs):
    t1 = time.time()
    cnnmodel.train()       # set the model to training mode
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        preds = cnnmodel(inputs)
        if use_cuda:
            preds = preds.cuda()

        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(cnnmodel.state_dict(), "molcnn_epoch{}.pt".format(epoch))

    cnnmodel.eval()        # set the model to evaluation mode
    eval_acc = evaluate_auc(cnnmodel, x_val, y_val)
    train_auc = evaluate_auc(cnnmodel, x_train, y_train)
    print('[epoch: {:d}] train_loss: {:.3f} train_auc: {:.3f}  val auc: {:.3f} time:  {}'.format(epoch, loss.data.item(), train_auc, eval_acc,  time.time()-t1))

# 0.77