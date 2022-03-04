#%%
%matplotlib inline 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score

df1 = pd.read_csv('classification_table.csv')
df2 = pd.read_csv('/home/phuijse/WORK/FUSA/training_datasets/datasets/VitGlobal/meta/audios_eruido2022_20220121.csv')
df1 = df1[["filename", "prediction_str"]]
df2 = df2[["WAVE_FILE", "WAVE_MEMO"]]
df1["filename"] = df1["filename"].apply(lambda x: int(x.split('_')[1]))
df3 = pd.merge(df1, df2, left_on='filename', right_on='WAVE_FILE')[["prediction_str", "WAVE_MEMO"]]
# %%
#mask_vitglobal = df3["WAVE_MEMO"].value_counts() >= 20
#labels = mask_vitglobal[mask_vitglobal].index
labels = ["GAVIOTA", "APILAR", "SIRENA", "BOCINA", "MOTO", "APILAR MOTOR"]
conf_matrix = pd.crosstab(df3['prediction_str'], df3['WAVE_MEMO'])
conf_matrix = conf_matrix[labels]
mask_fusa = conf_matrix[labels].T.sum() < 10
mask_fusa = mask_fusa[mask_fusa].index
conf_matrix.drop(mask_fusa, axis=0, inplace=True)

fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
data = (conf_matrix/conf_matrix.sum()).values
im = ax.imshow(data, cmap=plt.cm.Blues)
ax.set_yticks(np.arange(len(conf_matrix.index)), labels=conf_matrix.index)
ax.set_xticks(np.arange(len(labels)), labels=labels)
#cbar = ax.figure.colorbar(im, ax=ax)
kw = dict(horizontalalignment="center", verticalalignment="center")
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
        text = im.axes.text(j, i, f"{100*data[i, j]:0.0f}%", **kw)
        #texts.append(text)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor");
# %%
