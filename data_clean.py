import sys
import os
from zipfile import ZipFile
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
from sodapy import Socrata
import re
from tqdm.notebook import tqdm
import warnings
from datetime import datetime, timedelta
from urllib.request import urlopen
warnings.filterwarnings('ignore')

### Load/Retrieve all data

#### Retrieve data from API

## DOB Job Application Filings
## https://data.cityofnewyork.us/Housing-Development/DOB-Job-Application-Filings/ic3t-wcy2/data
data_set='ic3t-wcy2'
data_url='data.cityofnewyork.us'
# NYC Data API key - input key as 'app_token'
#from config import app_token
#app_token=app_token
app_token = '0h1v8vN3cR81KItbjfjYgRrAH'

client = Socrata(data_url,app_token)
# columns to retrieve from the dataset
cols = 'job__, doc__, house__, street_name, job_type, block, lot, bin__, borough, latest_action_date, pre__filing_date, owner_s_first_name, owner_s_last_name, owner_s_business_name'
results = client.get(data_set, job_type="NB", select = cols, limit=200000)
df = pd.json_normalize(results)

## Filter filing data to 2020 NBs
df = df[df["job_type"]=="NB"]
df.latest_action_date = pd.to_datetime(df.latest_action_date)
df.pre__filing_date = pd.to_datetime(df.pre__filing_date)
df['year'] = pd.DatetimeIndex(df['pre__filing_date']).year
df = df[df['year']==2021]
df = df[df['doc__']=="01"]

df.to_csv("raw-data/NB-2021.csv", index=False)

today = datetime.today()
s = today.strftime("%Y/%m/%d")
date = datetime.strptime(s, "%Y/%m/%d")
modified_date = date - timedelta(days=365*3)
back_to_date = datetime.strftime(modified_date, "%Y-%m-%d")

where_var = "good_through_date >" + ' "' + back_to_date + '"'

## ACRIS - Real Property Legals
## https://data.cityofnewyork.us/City-Government/ACRIS-Real-Property-Legals/8h5j-fqxa/data
data_url='data.cityofnewyork.us'
data_set='8h5j-fqxa'
client = Socrata(data_url,app_token)
client.timeout = 300
results_legals = client.get(data_set, where= where_var, select = 'borough, block, lot, street_number, street_name, document_id', limit=2000000)
rpl = pd.json_normalize(results_legals)

rpl.to_csv("raw-data/real_prop_legals.csv", index=False)

## ACRIS - Real Property Parties
## https://data.cityofnewyork.us/City-Government/ACRIS-Real-Property-Parties/636b-3b5g/data
data_set='636b-3b5g'
data_url='data.cityofnewyork.us'
client = Socrata(data_url,app_token)
client.timeout = 300
results = client.get(data_set, where= where_var, select = 'good_through_date, document_id, name, party_type',limit=43000000)
rpp = pd.json_normalize(results)

rpp.to_csv("raw-data/real_prop_parties.csv", index=False)

## ACRIS - Real Property Master
## https://data.cityofnewyork.us/City-Government/ACRIS-Real-Property-Master/bnx9-e6tj/data
data_url='data.cityofnewyork.us'
data_set='bnx9-e6tj'
client.timeout = 300
results_rpm = client.get(data_set,where= where_var, select = 'document_date, doc_type, document_id', limit=16000000)
rpm = pd.json_normalize(results_rpm)

rpm.to_csv("raw-data/real_prop_master.csv", index=False)

## ACRIS - Document Control Codes
## https://data.cityofnewyork.us/City-Government/ACRIS-Document-Control-Codes/7isb-wh4c/data
data_url='data.cityofnewyork.us'
data_set='7isb-wh4c'
client.timeout = 300
results_dcc = client.get(data_set, limit=150)
dcc = pd.json_normalize(results_dcc)

dcc.to_csv("raw-data/document_control_codes.csv", index=False)

### saved data from APIs
df = pd.read_csv("raw-data/NB-2021.csv")
rpl = pd.read_csv("./raw-data/real_prop_legals.csv")
rpp = pd.read_csv("raw-data/real_prop_parties.csv")
rpm = pd.read_csv("./raw-data/real_prop_master.csv")
dcc = pd.read_csv("raw-data/document_control_codes.csv")

### Clean data

def clean_data(nb_filing, rpp, rpl, rpm, dcc):
    '''
    cleans the datasets necessary for the model
    Keyword arguments:
    nb_filing -- DOB Job Application Filings (NYC OpenData) dataframe
    rpp -- ACRIS - Real Property Parties (NYC OpenData) dataframe
    rpl -- ACRIS - Real Property Legals (NYC OpenData) dataframe
    rpm -- ACRIS - Real Property Master (NYC OpenData) dataframe
    dcc -- ACRIS - Document Control Codes (NYC OpenData) dataframe
    '''

    # Add BBL code
    nb_filing['block'] = nb_filing.block.astype(int).astype(str)
    nb_filing['lot'] = nb_filing.lot.astype(int).astype(str)
    nb_filing['borough_code'] = 0
    nb_filing.loc[nb_filing['borough']=="MANHATTAN", 'borough_code'] = 1
    nb_filing.loc[nb_filing['borough']=="BRONX", 'borough_code'] = 2
    nb_filing.loc[nb_filing['borough']=="BROOKLYN", 'borough_code'] = 3
    nb_filing.loc[nb_filing['borough']=="QUEENS", 'borough_code'] = 4
    nb_filing.loc[nb_filing['borough']=="STATEN ISLAND", 'borough_code'] = 5
    nb_filing['BBL'] = nb_filing['borough_code'].astype(str) + nb_filing['block'].astype(str).str.zfill(5) + nb_filing['lot'].astype(str).str.zfill(4)
    rpl['BBL'] = rpl['borough'].astype(str) + rpl['block'].astype(str).str.zfill(5) + rpl['lot'].astype(str).str.zfill(4)

    # convert to date
    rpp['date'] = pd.to_datetime(rpp.good_through_date)
    rpp = rpp.sort_values('date', ascending=False)

    # Remove leading and ending whitespaces
    nb_filing.owner_s_business_name = [str(name).strip() for name in nb_filing.owner_s_business_name]
    nb_filing.house__ = [str(house_num).strip() for house_num in nb_filing.house__]
    nb_filing.street_name = [str(name).strip() for name in nb_filing.street_name]
    nb_filing.owner_s_first_name = [str(name).strip() for name in nb_filing.owner_s_first_name]
    nb_filing.owner_s_last_name = [str(name).strip() for name in nb_filing.owner_s_last_name]

    rpp.name = [str(name).strip() for name in rpp.name]

    rpl.street_number = [str(name).strip() for name in rpl.street_number]
    rpl.street_name = [str(name).strip() for name in rpl.street_name]

    rpp.document_id = [name.strip() for name in rpp.document_id]
    rpl.document_id = [name.strip() for name in rpl.document_id]
    rpm.document_id = [name.strip() for name in rpm.document_id]

    # add columns
    nb_filing['name'] = nb_filing['owner_s_first_name'] + " " + nb_filing['owner_s_last_name']
    nb_filing["NB_ADDRESS"] = nb_filing["house__"].map(str) + ' ' + nb_filing["street_name"].map(str)

    # fix N/A in nb_filing datset
    conditions = [(nb_filing.owner_s_business_name == "N/A"),
                  (nb_filing.owner_s_business_name.isna()),
                  (nb_filing.owner_s_business_name != "N/A")]

    choices = [nb_filing.name, nb_filing.name, nb_filing.owner_s_business_name]
    nb_filing["owner_s_business_name"] = np.select(conditions, choices)

    # drop duplicate BBLs
    nb_filing = df.drop_duplicates("BBL")

    # subset datasets
    rpm = rpm[["document_id", "doc_type", "document_date"]]
    dcc = dcc.rename(columns={"doc__type": "doc_type"})
    # add Document Control Codes to Real Property Masters
    rpm = pd.merge(rpm, dcc, on="doc_type", how="left")

    # clean up date
    rpm['doc_date'] = pd.to_datetime(rpm.document_date, errors = 'coerce')

    # rearrange columns so date columns are by each other
    rpm = rpm[['document_id','doc_type','document_date', 'doc_date','record_type','doc__type_description',
     'class_code_description','party1_type','party2_type','party3_type']]

    return(nb_filing, rpp, rpl, rpm, dcc)

df, rpp, rpl, rpm, dcc = clean_data(df, rpp, rpl, rpm, dcc)
