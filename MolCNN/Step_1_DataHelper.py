'''
Step 1: Generate data which is suitable for MolCNN

Input: 
    training_smiles.csv with data like this:
    
    INDEX,SMILES,ACTIVE
    1,Cl.O=C1C(CN(CC2:C:C:C:C:C:2)CC2:C:C:C:C:C:2)CCCC1CN(CC1:C:C:C:C:C:1)CC1:C:C:C:C:C:1,0
    2,CCOC(=O)C1OC2:C:C:C([N+](=O)[O-]):C:C:2C1(C)O,0
    3,COC1:C:C:C:C2:C:1OC1(N3CCOCC3)CCCCC1C2C[N+](=O)[O-],0
  
    test_smiles.csv with data like this: (without labels)
    
    INDEX,SMILES
    1,Cl.O=C1C(CN(CC2:C:C:C:C:C:2)CC2:C:C:C:C:C:2)CCCC1CN(CC1:C:C:C:C:C:1)CC1:C:C:C:C:C:1
    2,CCOC(=O)C1OC2:C:C:C([N+](=O)[O-]):C:C:2C1(C)O
    3,COC1:C:C:C:C2:C:1OC1(N3CCOCC3)CCCCC1C2C[N+](=O)[O-]


Output:
    Dict: cnndata.p
    
    alldatadict = {'train': data,'y': y, 'test':test ,'vocabs':vocabs, 
                   'embed_size':embed_size, 'weight':weight,
                   'idx_to_word':idx_to_word, 'word_to_idx':word_to_idx}
'''

import warnings
warnings.filterwarnings('ignore')#处理警告

import pandas as pd
import numpy as np
from rdkit import Chem
from mol2vec.features import mol2alt_sentence,MolSentence
from gensim.models import word2vec
import torch


data = pd.read_csv("../training_smiles.csv")
y = np.array(data["ACTIVE"].astype(int))

data = data[["SMILES"]]
data["SMILES_str"] = data["SMILES"] 
data["SMILES"] = data["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
model = word2vec.Word2Vec.load('../models/model_300dim.pkl')
data['sentence'] = data.apply(lambda x: MolSentence(mol2alt_sentence(x['SMILES'], 1)), axis=1)
data = [x.sentence for x in data['sentence']]


vocabs = [x for x in model.wv.index2word if x != 'UNK']
vocab_size = len(vocabs)+1
embed_size = model.wv.vector_size
weight = torch.zeros(vocab_size, embed_size)
word_to_idx = {word: i+1 for i, word in enumerate(vocabs) }
word_to_idx['UNK']=0
idx_to_word = {i+1: word for i, word in enumerate(vocabs) }
idx_to_word[0]='UNK'
vocabs.append('UNK')
for i in range(len(vocabs)):
    index = word_to_idx[vocabs[i]]
    weight[index, :] = torch.from_numpy(model.wv.get_vector(idx_to_word[word_to_idx[vocabs[i]]]))

data = [ [word_to_idx[x]  if x in word_to_idx else word_to_idx['UNK'] for x in sents] for sents in data]
data = pd.DataFrame(data)
data = data.fillna(0)

alldatadict = {'train': data, 'vocabs':vocabs, 'embed_size':embed_size, 'weight':weight,
               'idx_to_word':idx_to_word, 'word_to_idx':word_to_idx}

maxsentlen = 444

test = pd.read_csv("../test_smiles.csv")
test = test[["SMILES"]]
test["SMILES_str"] = test["SMILES"] 
test["SMILES"] = test["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
test['sentence'] = test.apply(lambda x: MolSentence(mol2alt_sentence(x['SMILES'], 1)), axis=1)
test = [x.sentence for x in test['sentence']]


test = [ [word_to_idx[x]  if x in word_to_idx else word_to_idx['UNK'] for x in sents] for sents in test]

test = pd.DataFrame(test)
padding = np.zeros([len(test), maxsentlen-test.shape[1]])
test = pd.concat([test, pd.DataFrame(padding)], axis=1)
test.columns = [i for i in range(test.shape[1])]
test = test.fillna(0)

alldatadict = {'train': data,'y': y, 'test':test ,'vocabs':vocabs, 'embed_size':embed_size, 'weight':weight,
               'idx_to_word':idx_to_word, 'word_to_idx':word_to_idx}

import pickle

pickle.dump(alldatadict, open( "cnndata.p", "wb" ) )

