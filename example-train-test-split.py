#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


filenames= ['CIRCLE_seq_10gRNA_wholeDataset', 'SITE-Seq_offTarget_wholeDataset']
file_column_dict = {'CIRCLE_seq_10gRNA_wholeDataset':('sgRNA_seq', 'off_seq', 'label', 'sgRNA_type'),
                    'SITE-Seq_offTarget_wholeDataset':('on_seq', 'off_seq', 'reads', 'on_seq')}


# In[3]:


def load_data(filename):
    columns = file_column_dict[filename]
    data = pd.read_csv('datas/{}.csv'.format(filename))
    sgRNAs = data[columns[0]]
    DNAs = data[columns[1]]
    labels = data[columns[2]]
    sgRNA_types = data[columns[3]]
    sgRNAs = sgRNAs.apply(lambda sgRNA: sgRNA.upper())
    DNAs = DNAs.apply(lambda DNA: DNA.upper())
    labels = labels.apply(lambda label: int(label!=0))
    sgRNAs_new = []
    for index, sgRNA in enumerate(sgRNAs):
        sgRNA = list(sgRNA)
        sgRNA[-3] = DNAs[index][-3]
        sgRNAs_new.append(''.join(sgRNA))
    sgRNAs = pd.Series(sgRNAs_new)
    data = pd.DataFrame.from_dict({'sgRNAs':sgRNAs, 'DNAs':DNAs, 'labels':labels})
    return data[data.apply(lambda row: 'N' not in list(row['DNAs']), axis = 1)]


# In[4]:


datas = []
for filename in filenames:
    data = load_data(filename)
    datas.append(data)
datas = pd.concat(datas, axis=0)
train, test = train_test_split(datas, test_size=0.2, random_state=42)


# In[5]:


train.to_csv('example_saved/example-train-data.csv', index=False)
test.to_csv('example_saved/example-test-data.csv', index=False)


# In[ ]:




