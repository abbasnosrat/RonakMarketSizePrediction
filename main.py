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
         Hover your cursor on the ? if you want information on each component. Also, the documentation is available on [this Google doc](https://docs.google.com/document/d/1oMk5kQi6FAgqsGGXW-ksRVP8OyhvmnbUnxn0mpi5x2U/edit?usp=sharing). You can find a detailed guide of the app on [this doc](https://docs.google.com/document/d/1J3bzPC_u5nAXrmgdaiQtL9J35yV_dVR7XLDImyE_78Y/edit?usp=sharing)
         """)
st.sidebar.write("Controls")
sheet_id = "1PNTC8IvqruHs3DWVX6HW30d2TCM6z3PCxtRMA_qep0M"
sheet_name = "Sheet1"
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
helps = pd.read_csv(url,index_col=0)
file = st.sidebar.file_uploader("Upload Your Dataset", type=".csv",help=helps.loc["Upload Your Dataset"].Description)
use_sample_data = st.sidebar.checkbox("Use Sample Data",
                                      help=helps.loc["Use Sample Data"].Description)

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
                                    help=helps.loc["Please Select A Product"].Description)
    df_t = df.query(f"GenericName == '{product}'").reset_index(drop=True)
    
   
    pack_check = st.sidebar.checkbox("display in packs", help="displays outputs in ronak's pack")
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
                                    help=helps.loc["Select Prediction Horizon"].Description))
    test_size_manual = st.sidebar.number_input(label="Select Test Size", min_value=0, max_value=30, value=0,
                                               help=helps.loc["Select Test Size"].Description)
    manual = st.sidebar.checkbox("Manual Mode", help=helps.loc["Manual Mode"].Description)
    

    


    train_size = -5 if test_size_manual == 0 else -test_size_manual
    model_name = st.selectbox("select your model", options=["XGB", "Prophet"])
    if model_name == "XGB":
       df_final, dg =  XGB(manual, df_t, train_size, horizon, helps)
       df_final["Year"] = df_final["Date"].apply(lambda d: d.split(("/"))[0])
       df_final["Month"] = df_final["Date"].apply(lambda d: d.split(("/"))[1])
       df_final = df_final.drop(columns="Date")
       df_final = df_final.rename(columns={"Yhat":"y"})
       cols = ["Year", "Month", "y"]
       total_df = pd.concat([dg[cols], df_final[cols]],axis=0)
    else:
       df_final, dg =  FBProphet(manual, df_t,test_size_manual, horizon, helps)
       df_final["Year"] = df_final["Date"].apply(lambda d: int(d.split(("/"))[0]))
       df_final["Month"] = df_final["Date"].apply(lambda d: int(d.split(("/"))[1]))
       df_final = df_final.drop(columns="Date")
       df_final = df_final.rename(columns={"Yhat":"y"})
       dg["Year"] = dg["ds"].apply(lambda d: int(d.split(("/"))[0]))
       dg["Month"] = dg["ds"].apply(lambda d: int(d.split(("/"))[1]))
       dg = dg.drop(columns="ds")
       cols = ["Year", "Month", "y"]
       total_df = pd.concat([dg[cols], df_final[cols]],axis=0)

    total_df = total_df.sort_values(["Year", "Month"])
    agg_months = st.sidebar.number_input(label="Select Aggregation Months", help="Select how many months you want to be aggregated",value=12)

    st.write(f'## Sum of sales for the selected period is {int(np.round(total_df.tail(agg_months)["y"].sum()))}')
    # plt.plot(preds, label="prediction")
    # plt.savefig("ar_pred.jpg")
else:
    
    st.write("Please upload your data")
    # df = pd.read_csv("SalesData.csv")[["GoodName", "StrFactDate", "SaleAmount"]]
    # csv = convert_df(df)
    # st.download_button("Sample Data", csv, "SampleData.csv","text/csv",
    # key='download-csv')