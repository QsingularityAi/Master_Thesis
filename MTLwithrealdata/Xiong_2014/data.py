import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from utils import train_test_split_for_array, numpy_to_pytorch

def data_prepration():
    
    data = pd.read_csv('/Users/anuragtrivedi/Desktop/MTLwithrealdata/Xiong_2014/data.csv')
    df_new = pd.DataFrame(data)
    
#df_new.skew()
    
    numeric_cols = df_new.select_dtypes(include=['int64', 'float64']).columns.tolist()
    imputer1 = SimpleImputer(strategy = 'mean')
    imputer1.fit(df_new[numeric_cols])
    df_new[numeric_cols] = imputer1.transform(df_new[numeric_cols])
    df_new[numeric_cols].describe().loc[['min', 'max']]

    scaler1 = MinMaxScaler()
    scaler1.fit(df_new[numeric_cols])
    df_new[numeric_cols] = scaler1.transform(df_new[numeric_cols])
    df_new[numeric_cols].describe().loc[['min', 'max']]
    
    # split data in to feature and target lable
    input_cols = list(df_new.columns)[0:4]
    target_cols1 = list(df_new.columns)[4:5]
    target_cols2 = list(df_new.columns)[5:6]
    
    inputs_df = df_new[input_cols].copy()
    target_df1 = df_new[target_cols1].copy()
    target_df2 = df_new[target_cols2].copy()

# Convert to numpy_array
    X = inputs_df.to_numpy()
    Y1 = target_df1.to_numpy()
    Y2 = target_df2.to_numpy()

# split data in to train, val and test    
       
    X, X_test, Y1, Y1_test, Y2, Y2_test = train_test_split_for_array(X, Y1, Y2,
                                                                     test_size=0.10, 
                                                                     random_state=42)
    # split data in to train, and val    
    X_train, X_valid, Y1_train, Y1_valid, Y2_train, Y2_valid = train_test_split_for_array(X, Y1, Y2,
                                                                                         test_size=0.10, 
                                                                                         random_state=42)
# Convert to numpy_array to torch tensor
    X_train, Y1_train, Y2_train = numpy_to_pytorch(X_train, Y1_train, Y2_train)
    X_valid, Y1_valid, Y2_valid = numpy_to_pytorch(X_valid, Y1_valid, Y2_valid)
    X_test, Y1_test, Y2_test = numpy_to_pytorch(X_test, Y1_test, Y2_test)
    return X_train, Y1_train, Y2_train, X_valid, Y1_valid, Y2_valid, X_test, Y1_test, Y2_test