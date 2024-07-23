import datetime
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jdatetime
from matplotlib.style import use
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from tqdm.auto import trange
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_percentage_error

from utils import *












st.write("""
         # Market Size Prediction
         Hover your curser on the ? if you want information on each component
         """)
st.sidebar.write("Controls")
file = st.sidebar.file_uploader("Upload Your Dataset", type=".csv",help="You can upload the data you want the model to be trained on")
use_sample_data = st.sidebar.checkbox("Use Sample Data",
                                      help="Check this if you do not want to upload a dataset and want to upload data and the model will be trained on the sample dataset")

# df = pd.read_csv("SalesData.csv") if file is None else pd.read_csv(file)
try:
    df = pd.read_csv(file)
    got_data = True
except:
    if use_sample_data:
        df = pd.read_csv("./stat.csv") 
        got_data = True
    else:
        got_data = False

if got_data:
    pack_map = df[["GenericName", "CompanyName", "num_in_pack"]].query("CompanyName == 'روناک'").drop_duplicates(ignore_index=True)
    df = df[["GenericName", "CompanyName", "Year", "Month","QtyAdadi"]]
    products = list(df.GenericName.unique())
    product = st.sidebar.selectbox(label="Please select a Generic", options=products,
                                    help="Select the product you want the model to predict. Keep in mind that the model cannot be trained on a product with low data.")
    df_t = df.query(f"GenericName == '{product}'").reset_index(drop=True)
    
   
    pack_check = st.checkbox("display in packs", help="displays outputs in ronak's pack")
    if pack_check:
        pack_map_t = pack_map.query(f"GenericName == '{product}'")
        if len(pack_map_t)>0:
            pack = pack_map_t.reset_index(drop=True).iloc[0,-1]
            df_t["QtyAdadi"] = df_t["QtyAdadi"]/pack
            st.write(f"ronak's num in pack = {pack}")
    company_level = st.sidebar.checkbox("Single Company", help="check this box if you want to select a company")
    if company_level:
        companies = list(df_t.CompanyName.unique())
        company =  st.sidebar.selectbox(label="Please select a Company", options=companies,
                                    help="Select the Company you want the model to predict. Keep in mind that the model cannot be trained on a product with low data.")
        df_t = df_t.query(f"CompanyName == '{company}'").sort_values(["Year", "Month"], ignore_index=True)[["Year", "Month","QtyAdadi"]]
        
    else:
        df_t = df_t.groupby(["Year", "Month"]).agg({"QtyAdadi":"sum"}).reset_index().sort_values(["Year", "Month"], ignore_index=True)
        
        
    horizon = int(st.sidebar.slider(label="Select Prediction Horizon", min_value=2, max_value=30, value=5,
                                    help="You can select how many months do you want the model to predict into the future."))
    test_size_manual = st.sidebar.number_input(label="Select Test Size", min_value=0, max_value=30, value=0,
                                               help="""The data is divided into training and testing datasets, these datasets are used for tuning the model parameters.
                                               Sometimes, the test data may suffer a trend change, which is not present in the train data.
                                               For example, the trend is increasing or flat in the training data and it changes to a declining trend after the split.
                                               Consequently, the model is not prepared for this change, which leads to poor predictions on the test data.
                                               To mitigate this issue, you can change how many months are kept as test data. By default, the application keeps 10 months for testing
                                               if the dataset has more than 20 months and 2 months for if the dataset has less than 20 months of data. The default values are selected if this input's value is zero.""")
    manual = st.sidebar.checkbox("Manual Mode", help='''The model uses Bayesian optimization for hyper-parameter tuning.
                                 This process is time consuming and sometimes, it may not find the optimal parameters.
                                 By checking this box, you can bypass the automatic tuning and select the hyper-parameters manually.''')
    

    



    train_size = -5 if test_size_manual == 0 else -test_size_manual
    model_name = st.selectbox("select your model", options=["XGB", "Prophet"])
    if model_name == "XGB":
        XGB(manual, df_t, train_size, horizon)
    else:
        FBProphet(manual, df_t,test_size_manual, horizon)

    # plt.plot(preds, label="prediction")
    # plt.savefig("ar_pred.jpg")
else:
    
    st.write("Please upload your data")
    # df = pd.read_csv("SalesData.csv")[["GoodName", "StrFactDate", "SaleAmount"]]
    # csv = convert_df(df)
    # st.download_button("Sample Data", csv, "SampleData.csv","text/csv",
    # key='download-csv')