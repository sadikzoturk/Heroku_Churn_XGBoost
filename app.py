#!/usr/bin/env python
# coding: utf-8


from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import json
import pickle


import sklearn  
from sklearn import preprocessing



app = Flask(__name__)


main_cols = pickle.load(open("columns.pkl", 'rb'))





def clean_data(df_x):
    le = preprocessing.LabelEncoder()
    df_x.Gender = le.fit_transform(df_x.Gender)
    df_x = pd.get_dummies(data = df_x,  columns=["Geography"], drop_first = False)
    return df_x




def standardize_data(dta):
                        
    scaler = pickle.load(open("std_scaler.pkl", 'rb'))
    X_transformed = scaler.transform(dta)
    return X_transformed





@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])

def predict():
    
    form_data = request.form.to_dict()
    print("form_data yazdırılıyor ******************************")
    print(form_data)
    df_input = pd.DataFrame.from_records([form_data])
    #df_input = df_input.drop(['submitBtn'], axis=1)
    df_input = pd.DataFrame(df_input)
    
    sample_df = pd.DataFrame(columns = main_cols)
    clean_df = clean_data(df_input)
    main_df = sample_df.append(clean_df,sort=False)
    main_df = main_df.fillna(0)
    print(main_df)





    std_df = standardize_data(main_df)
    print("std_df yazdırılıyor ******************************")
    print(std_df)
    
    clf = pickle.load(open('model.pkl', 'rb'))
    pred = clf.predict_proba(std_df)

    print("pred yazdırılıyor ******************************")
    print(pred)
    #x = round(pred*100, 2)
    x = pred[0]*100


    return render_template('index.html', predicted_value="XGBoost - Customer Churn rate: {}".format(x))
    


if __name__ == '__main__':
    app.run(debug=True)







