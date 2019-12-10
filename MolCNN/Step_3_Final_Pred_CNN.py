import warnings
warnings.filterwarnings('ignore')#处理警告
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


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


alldatadict=pickle.load(open( "cnndata.p", "rb" ) )

weight = alldatadict['weight']
use_cuda = torch.cuda.is_available()

loadmodel = TextCNN(max_sent_len=444,
                    embedding_dim=300,
                    filter_sizes=[3,4,5],
                    num_filters=100,
                    vocab_size=21003,
                    num_classes=2)
loadmodel.load_state_dict(torch.load("molcnn.pt"))

test = alldatadict['test'].values
xt =  torch.from_numpy(test).long()
inputs = autograd.Variable(xt)
preds = loadmodel(inputs)
preds = preds[:,1].data
predt = preds.data.numpy()

output = np.concatenate(([0.77], predt)) 
np.savetxt('cnn_predition.txt', output)

