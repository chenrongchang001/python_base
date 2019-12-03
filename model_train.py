import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

mutations = pd.read_csv('74.samples.mutations.csv',sep=',')
levels = pd.read_csv('31.sample.verify.pos',sep='\t', names=['Sample','Chromosome','End','Ref','Alt','Levels'])
mutations_levels = pd.merge(mutations, levels, on=['Sample','Chromosome','End','Ref','Alt'])
mutations_levels = mutations_levels.loc[mutations_levels.Levels.isin(['A', 'C']),:]

mutations_levels.loc[mutations_levels.Levels == 'A','Levels'] = 1
#mutations_levels.loc[mutations_levels.Levels == 'B','Levels'] = 1
mutations_levels.loc[mutations_levels.Levels == 'C','Levels'] = 0

def get_data(mutations_levels):
    data = pd.DataFrame()
    data['Vaf'] = mutations_levels.Alt_Depth/mutations_levels.Depth
    data['Third_Base'] = mutations_levels.Third_Base_Vaf
    data['Third_Base_Quality'] = mutations_levels.Third_Base_Average_Quality
    data['Delete_Base'] =  mutations_levels.Delete_Vaf
    data['Strandness'] =  mutations_levels.Alt_Plus/(mutations_levels.Alt_Plus + mutations_levels.Alt_Minus)
    data['Alt_Position'] = mutations_levels.Alt_Position
    data['Alt_Base_Quality'] = mutations_levels.Alt_Base_Quality
    data['Alt_Map_Quality'] = mutations_levels.Alt_Map_Quality
    data['Alt_Average_Mismatch_Count'] = mutations_levels.Alt_Average_Mismatch_Count
    data['Alt_Average_Mismatch_Base_Quality'] = mutations_levels.Alt_Average_Mismatch_Base_Quality
    data['Alt_Average_Mismatch_Base_Distance_To_Mut'] = mutations_levels.Alt_Average_Mismatch_Base_Distance_To_Mut
    data['Alt_Mismatch_Base_Reference_Position_Variation'] = mutations_levels.Alt_Mismatch_Base_Reference_Position_Variation
    data['Alt_Mismatch_Base_Nearby'] = mutations_levels.Alt_Mismatch_Base_Nearby
    data['Alt_Indel_Count'] = (mutations_levels.Alt_Insert_Count + mutations_levels.Alt_Delete_Count)
    data['Ref_Base_Quality'] = mutations_levels.Ref_Base_Quality
    data['Ref_Map_Quality'] = mutations_levels.Ref_Map_Quality
    data['Ref_Average_Mismatch_Count'] = mutations_levels.Ref_Average_Mismatch_Count
    data['Ref_Average_Mismatch_Base_Quality'] = mutations_levels.Ref_Average_Mismatch_Base_Quality
    data['Ref_Average_Mismatch_Base_Distance_To_Mut'] = mutations_levels.Ref_Average_Mismatch_Base_Distance_To_Mut
    data['Ref_Mismatch_Base_Reference_Position_Variation'] = mutations_levels.Ref_Mismatch_Base_Reference_Position_Variation
    data['Ref_Mismatch_Base_Nearby'] = mutations_levels.Ref_Mismatch_Base_Nearby
    data['Ref_Indel_Count'] = (mutations_levels.Ref_Insert_Count + mutations_levels.Ref_Delete_Count)
    data['Depth_in_normal'] = mutations_levels.Depth_in_normal
    data['Vaf_in_normal'] = mutations_levels.Vaf_in_normal
    data['Base_Quality_Diff'] = data['Ref_Base_Quality'] - data['Alt_Base_Quality']
    data['Map_Quality_Diff'] = data['Ref_Map_Quality'] - data['Alt_Map_Quality']
    data['Average_Mismatch_Count_Diff'] = data['Alt_Average_Mismatch_Count'] - data['Ref_Average_Mismatch_Count']
    data['Average_Mismatch_Base_Quality_Diff'] = data['Alt_Average_Mismatch_Base_Quality'] - data['Ref_Average_Mismatch_Base_Quality']
    data['Indel_Count_Diff'] = data['Alt_Indel_Count'] - data['Ref_Indel_Count']
    data = data.values
    return data

data = get_data(mutations_levels)
target = to_categorical(mutations_levels.Levels.values,num_classes=2)

train_data, test_data, train_target, test_target = train_test_split(data, target,test_size = 0.3,stratify = target)

model = Sequential()
model.add(Dense(30, input_dim=29, kernel_initializer='uniform', activation='relu')) 
model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(2, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(train_data, train_target, epochs=80, batch_size=20)
# evaluate the model
scores = model.evaluate(test_data, test_target)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('modelV1.0.1')
