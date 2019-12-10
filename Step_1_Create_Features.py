'''
Step 1: Create Features

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
    Dict: datadict
    datadict = {"Morgan": finger, "Despcritor": traindata, "molvec": m2v, 'y': y}
    
    datadict['Morgan'] -> Morgan Fingerprints nbits=512
    datadict['Despcritor'] -> Chemical Despcritors 
    datadict['molvec'] -> Mol2vec dim=300

'''

import warnings
warnings.filterwarnings('ignore')#处理警告

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  
import rdkit.Chem.Fragments as f
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
from rdkit.Chem import AllChem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec


##########  Feature extraction

def create_features(data, types="train"):

    if types == "train":
        y = np.array(data['ACTIVE'].astype(int))  
    elif types == "test":
        y = None
    
    data = data[["SMILES"]]
    data["SMILES_str"] = data["SMILES"] 
    data["SMILES"] = data["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    data["NumAtoms"] = data["SMILES"].apply(lambda x: x.GetNumAtoms()) #l.HeavyAtomCount(m)
    data["ExactMolWt"] = data["SMILES"].apply(lambda x: d.CalcExactMolWt(x))
    data["fr_Al_COO"] = data["SMILES"].apply(lambda x: f.fr_Al_COO(x))
    data["HsNumAtoms"] = data["SMILES"].apply(lambda x: Chem.AddHs(x).GetNumAtoms())
    #to have the hydrogens explicitly present
    
    BondType = [[str(x.GetBondType()) for x in m.GetBonds()] for m in data["SMILES"]]
    BondType = [" ".join(x) for x in BondType]
    
    vec = CountVectorizer().fit(BondType) 
    train_tfidf = vec.transform(BondType).todense()   # 转化为更直观的一般矩阵
    vocabulary = vec.vocabulary_
    
    train_tfidf = pd.DataFrame(train_tfidf)
    train_tfidf.columns = vocabulary
    
    data = pd.concat([data, train_tfidf], axis=1)
    #data.columns
    #['SMILES', 'ACTIVE', 'SMILES_str', 'NumAtoms', 'ExactMolWt', 'fr_Al_COO','HsNumAtoms', 'double', 'single', 'aromatic', 'triple']
    traindata=data[['NumAtoms', 'ExactMolWt', 'fr_Al_COO','HsNumAtoms', 'double', 'single', 'aromatic', 'triple']]


    finger = [np.array(AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=512)) for x in data["SMILES"]]
    finger = pd.DataFrame(finger)
    finger.columns = ["morgan_"+str(x) for x in finger.columns]

    model = word2vec.Word2Vec.load('models/model_300dim.pkl')
    data['sentence'] = data.apply(lambda x: MolSentence(mol2alt_sentence(x['SMILES'], 1)), axis=1)
    m2v = [DfVec(x) for x in sentences2vec(data['sentence'], model, unseen='UNK')]
    m2v = np.array([x.vec for x in m2v])
    m2v = pd.DataFrame(m2v)
    m2v.columns = ["m2v_"+str(x) for x in m2v.columns]
    
    datadict = {"Morgan": finger, "Despcritor": traindata, "molvec": m2v, 'y': y}

    return datadict

train = pd.read_csv("training_smiles.csv")
traindict = create_features(train, types="train")

test = pd.read_csv("test_smiles.csv")
testdict = create_features(test, types="test")

pickle.dump(traindict, open( "traindict.p", "wb" ) )
pickle.dump(testdict, open( "testdict.p", "wb" ) )