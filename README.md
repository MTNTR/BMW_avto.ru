# BMW_avto.ru
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from catboost import CatBoostRegressor
train = pd.read_csv('BMW_train_ANSI.csv', encoding = 'ANSI', sep ='|')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
data = train.copy()
s_drop = sample_submission.drop([823])
y_true = s_drop['price']
def preproc_data(df_input):
    data = df_input.copy()
    data['modelDate'] = 2020 - data['modelDate']
    data['productionDate'] = 2020 - data['productionDate']
    data['enginePower'] = data['enginePower'].str[:-4].apply(float)
    data = data[(data.engineDisplacement != 'undefined LTR') ]
    data['engineDisplacement'] = data['engineDisplacement'].str[:-4].apply(float)
    data['name'] = data['name'].str[:4]
    data['enginePower'] = data['enginePower']*10
    data['engineDisplacement'] = data['engineDisplacement']*10
    df_output = data[['bodyType', 'name' , 'color', 'vehicleTransmission' , 'fuelType' , 'Привод' , 
                 'modelDate' ,'productionDate' ,'mileage', 'engineDisplacement' , 'enginePower' , 'price']]
    df_output=df_output.dropna()
    for feature in ['modelDate', 'mileage', 'productionDate' , 'enginePower' ,'engineDisplacement']:
        df_output[feature]=df_output[feature].astype('int32')
    return df_output
def preproc_dataX(df_input):
    data = df_input.copy()
    data['modelDate'] = 2020 - data['modelDate']
    data['productionDate'] = 2020 - data['productionDate']
    data['enginePower'] = data['enginePower'].str[:-4].apply(float)
    data = data[(data.engineDisplacement != 'undefined LTR') ]
    data['engineDisplacement'] = data['engineDisplacement'].str[:-4].apply(float)
    data['name'] = data['name'].str[:4]
    data['enginePower'] = data['enginePower']*10
    data['engineDisplacement'] = data['engineDisplacement']*10
    df_output = data[['bodyType', 'name' , 'color','vehicleTransmission' , 'fuelType' , 'Привод' , 
                 'modelDate' ,'productionDate' ,'mileage', 'engineDisplacement' , 'enginePower']]
    df_output=df_output.dropna()
    for feature in ['modelDate', 'mileage', 'productionDate' , 'enginePower' ,'engineDisplacement']:
        df_output[feature]=df_output[feature].astype('int32')
    return df_output
train_preproc = preproc_data(train)
X_sub = preproc_dataX(test)
X = train_preproc.drop(['price'], axis=1,)
y = train_preproc.price.values
ITERATIONS = 2000
LR = 0.1
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 1000)[0].tolist()
def cat_model(y_train, X_train, X_test, y_test):

    model = CatBoostRegressor(iterations = 2000,
                              learning_rate = 0.1,
                              random_seed = 42,
                              eval_metric='MAPE',
                              custom_metric=['R2', 'MAE']
                             )
    model.fit(X_train, y_train,
             cat_features=cat_features_ids,
             eval_set=(X_test, y_test),
             verbose_eval=100,
             use_best_model=True,
             plot=True
             )
    return (model)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))
model = cat_model(y_train, X_train, X_test, y_test)
y_pred = model.predict(X_sub)
mape(y_true, y_pred)
