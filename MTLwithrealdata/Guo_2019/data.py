import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from utils import train_test_split_for_array, numpy_to_pytorch

def data_prepration():
    
    input_data = pd.read_excel('/Users/anuragtrivedi/Desktop/MTLwithrealdata/Guo_2019/data_x.xlsx', engine='openpyxl',  header=0)
    df = input_data.drop(['Unnamed: 0'], axis = 1)
    
# using SimpleImputer to fill Nan Value in Input col  
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    imputer1 = SimpleImputer(missing_values=np.NaN, strategy = 'mean')
    imputer1.fit(df[numeric_cols])
    df[numeric_cols] = imputer1.transform(df[numeric_cols])
# using SimpleImputer to fill Nan Value in Output col
    
    
# Input Data is positive and negative skewness 
    pt=PowerTransformer(method='yeo-johnson') 
    X_power=pt.fit_transform(df)
    df=pd.DataFrame(X_power,columns=df.columns)
    #df.skew()
# Input Data is normalized by minmaxscaler
    scaler1 = MinMaxScaler()
    scaler1.fit(df[numeric_cols])
    df[numeric_cols] = scaler1.transform(df[numeric_cols])

# split data in to feature and target lable
    input_cols = list(df.columns)[0:27]
    target_cols1 = list(df.columns)[27:28]
    target_cols2 = list(df.columns)[28:29]
    target_cols3 = list(df.columns)[29:30]
    
    inputs_df = df[input_cols].copy()
    target_df1 = df[target_cols1].copy()
    target_df2 = df[target_cols2].copy()
    target_df3 = df[target_cols3].copy()
# Convert to numpy_array
    X = inputs_df.to_numpy()
    Y1 = target_df1.to_numpy()
    Y2 = target_df2.to_numpy()
    Y3 = target_df3.to_numpy()

# split data in to train, val and test    
    X, X_test, Y1, Y1_test, Y2, Y2_test, Y3, Y3_test = train_test_split_for_array(X, Y1, Y2, Y3,
                                                                                      test_size=0.2, 
                                                                                     random_state=42)
    # split data in to train, val and test    
    X_train, X_valid, Y1_train, Y1_valid, Y2_train, Y2_valid, Y3_train, Y3_valid = train_test_split_for_array(X, Y1, Y2, Y3,
                                                                                      test_size=0.1, 
                                                                                     random_state=42)
# Convert to numpy_array to torch tensor
    X_train, Y1_train, Y2_train, Y3_train = numpy_to_pytorch(X_train, Y1_train, Y2_train, Y3_train)
    X_valid, Y1_valid, Y2_valid, Y3_valid = numpy_to_pytorch(X_valid, Y1_valid, Y2_valid, Y3_valid)
    X_test, Y1_test, Y2_test, Y3_test = numpy_to_pytorch(X_test, Y1_test, Y2_test, Y3_test)
    return X_train, Y1_train, Y2_train, Y3_train, X_valid, Y1_valid, Y2_valid, Y3_valid, X_test, Y1_test, Y2_test, Y3_test