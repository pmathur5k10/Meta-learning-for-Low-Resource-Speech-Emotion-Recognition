import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import numpy as np


# train_df = pd.read_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/train_emo.csv')
# test_df = pd.read_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/test_emo.csv')

#train_df = pd.read_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/ravdess_files/train_ravdess.csv')
#test_df = pd.read_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/ravdess_files/test_ravdess.csv')

#data = pd.concat([train_df,test_df])
#data.drop(data.columns[[0]], axis = 1, inplace = True) 

data = pd.read_csv('./iemocap.csv')

X = data.iloc[:,0]
Y = data.iloc[:,1]

#print ( np.unique(Y, return_counts = True) )
#sys.exit()

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=55)
train_index,tmp_index = next(iter(sss.split(X,Y)))

train_X = X.iloc[train_index]
train_Y = Y.iloc[train_index]


temp_X = X.iloc[tmp_index]
temp_Y = Y.iloc[tmp_index]

sss = StratifiedShuffleSplit(n_splits=1,test_size=(2/3), random_state=55)
val_index,test_index = next(iter(sss.split(temp_X,temp_Y)))


val_X = temp_X.iloc[val_index]
val_Y = temp_Y.iloc[val_index]

test_X = temp_X.iloc[test_index]
test_Y = temp_Y.iloc[test_index]

final_train = pd.concat([train_X,train_Y],axis=1)
final_val = pd.concat([val_X,val_Y],axis=1)
final_test = pd.concat([test_X,test_Y],axis=1)


tmp_train = train_X.tolist()
tmp_val = val_X.tolist()
tmp_test = test_X.tolist()


# final_train.to_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/train.csv',index = False)
# final_val.to_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/val.csv',index = False)
# final_test.to_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/test.csv',index = False)


#final_train.to_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/ravdess_files/train.csv',index = False)
#final_val.to_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/ravdess_files/val.csv',index = False)
#final_test.to_csv('/gdrive/My Drive/IS_Meta_Learning/datasets/ravdess_files/test.csv',index = False)

#final_train.to_csv('./SAVEE/savee_train.csv',index = False)
#final_val.to_csv('./SAVEE/savee_val.csv',index = False)
#final_test.to_csv('./SAVEE/savee_test.csv',index = False)

#final_train.to_csv('./emospeech/emospeech_train.csv',index = False)
#final_val.to_csv('./emospeech/emospeech_val.csv',index = False)
#final_test.to_csv('./emospeech/emospeech_test.csv',index = False)

#final_train.to_csv('./TESS/tess_train.csv',index = False)
#final_val.to_csv('./TESS/tess_val.csv',index = False)
#final_test.to_csv('./TESS/tess_test.csv',index = False)

#final_train.to_csv('./ravdess-emotional-speech-audio/ravdess_train.csv',index = False)
#final_val.to_csv('./ravdess-emotional-speech-audio/ravdess_val.csv',index = False)
#final_test.to_csv('./ravdess-emotional-speech-audio/ravdess_test.csv',index = False)

#final_train.to_csv('./emo-db/emodb_train.csv',index = False)
#final_val.to_csv('./emo-db/emodb_val.csv',index = False)
#final_test.to_csv('./emo-db/emodb_test.csv',index = False)

final_train.to_csv('IEMOCAP/iemocap_train.csv',index = False)
final_val.to_csv('IEMOCAP/iemocap_val.csv',index = False)
final_test.to_csv('IEMOCAP/iemocap_test.csv',index = False)

