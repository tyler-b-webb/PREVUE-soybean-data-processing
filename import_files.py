#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:06:45 2022

@author: epnzv
"""
import datetime as dt
import pandas as pd

from calendar import monthrange

from aggregation_config import(ABM_TABLE, BIG_CF_FILE, BLIZZARD_DIR, CF_2022_FILE,
                               CF_2023_FILE, CM_DIR, DATA_DIR, EFFECTIVE_DATE, H2H_DIR,
                               HISTORICAL_SRP, HISTORICAL_SUPPLY, KYNETIC_DATA,
                               KYNETIC_COLUMNS_TO_DROP, KYNETIC_COLUMN_NAMES,
                               PROD_LIST_23, PROD_LIST_24, SALES_2021, SALES_2022,
                               SALES_DIR, YEARLY_ABM_FIPS_MAP)
from merge import (merge_2021_sales_data_w_date)
from preprocess import(create_late_lagged_sales, create_prediction_set,
                       merge_2021_sales_data_impute_daily,
                       merge_2022_sales_data_impute_daily, merge_2023_D1MS)


def fips_to_abm_by_year(df):
    """Joins the abm feature to a dataset based on FIPS code using the yearly
    abm map.
    
    Keyword arguments:
        df -- the dataframe with a FIPS feature
    Returns:
        df_with_abm -- the dataframe with abm joined
    """
    # read in the fips/abm map and subset out the fips/abm columns
    fips_abm_map = pd.read_csv(YEARLY_ABM_FIPS_MAP)
    fips_abm_map = fips_abm_map[['fips', 'abm', 'year']]
    
    # merge with the dataframe on fips
    df_with_abm = df.merge(fips_abm_map, on=['fips', 'year'])
    
    # drop the fips feature
    df_with_abm = df_with_abm.drop(columns=['fips'])

    return df_with_abm


def import_2023_product_list():
    """Reads in the 2023 product list, for use in the prediction set.
    
    Keyword arguments:
        None
    Returns:
        prod_list_23 -- the product list data series
    """
    prod_list_23 = pd.read_csv(DATA_DIR + PROD_LIST_23)
    
    return prod_list_23


def merge_2021_sales_data(df, abm_Teamkey):
    """Merges the 2021 sales data.
    
    Keyword arguments:
        df -- the dataframe to concat onto
    Returns:
        df_merged
    """
    # read in the 2021 sales data
    sales_2021 = pd.read_csv(DATA_DIR + SALES_2021)
    
    # grab relevant columns
    sales_2021_subset = sales_2021[['Team', 'VARIETY', 'CY Net Sales',
                                    'Returns', 'Haulbacks', 'Replants', 'Orders']]
    
    # rename the columns
    sales_2021_subset = sales_2021_subset.rename(columns={'Team': 'TEAM_KEY',
                                                          'VARIETY': 'Variety_Name',
                                                          'CY Net Sales': 'nets_Q',
                                                          'Returns': 'return_Q_ret',
                                                          'Haulbacks': 'return_Q_haul',
                                                          'Replants': 'replant_Q'})
    
    # remove any variety names with "Empty"
    sales_2021_subset = sales_2021_subset[
            sales_2021_subset['Variety_Name'] != '(Empty)'].reset_index(drop=True)
        
    sales_2021_subset = sales_2021_subset.merge(abm_Teamkey, on=['TEAM_KEY'],
                                                how='left')
    
    sales_2021_subset = sales_2021_subset.drop(columns=['TEAM_KEY'])
    
    # do the same for the returns/haulbacks and set return_Q to be the sum of the quantities
    sales_2021_subset['return_Q_ret'] = sales_2021_subset['return_Q_ret'].str.replace(',','')
    sales_2021_subset['return_Q_ret'] = sales_2021_subset['return_Q_ret'].astype('float64')
    sales_2021_subset['return_Q_haul'] = sales_2021_subset['return_Q_haul'].str.replace(',','')
    sales_2021_subset['return_Q_haul'] = sales_2021_subset['return_Q_haul'].astype('float64')
    
    # do the same for nets Q and replants
    sales_2021_subset['nets_Q'] = sales_2021_subset['nets_Q'].str.replace(',', '')
    sales_2021_subset['nets_Q'] = sales_2021_subset['nets_Q'].astype('float64')
    sales_2021_subset['replant_Q'] = sales_2021_subset['replant_Q'].str.replace(',', '')
    sales_2021_subset['replant_Q'] = sales_2021_subset['replant_Q'].astype('float64')

    
    sales_2021_subset['return_Q'] = (
            sales_2021_subset['return_Q_haul'] + sales_2021_subset['return_Q_ret'])
    
    sales_2021_subset = sales_2021_subset.drop(columns=['return_Q_haul',
                                                        'return_Q_ret'])

    # group by product and abm
    sales_2021_agg = sales_2021_subset.groupby(
            by=['Variety_Name', 'abm'], as_index=False).sum()
    
    sales_2021_agg['year'] = '2021'
    
    # merge in the order data
    sales_2021_monthly = merge_2021_sales_data_w_date(df=sales_2021_agg,
                                                      abm_Teamkey=abm_Teamkey)    
    
        
    # merge the lagged parts
    df_w_lag = create_late_lagged_sales(df=sales_2021_monthly,
                                        full_df=df,
                                        year=2021)
    
    # concatenate with the main dataframe
    df_merged = pd.concat([df, df_w_lag])
    
    return df_merged


def merge_2022_sales_data(df, abm_Teamkey):
    """Merges in the 2022 sales data. This is from DSM, not SCM.
    
    Keyword arguments:
        df -- the dataframe we are going to merge the sales data with
        abm_Teamkey -- the team key to abm converter
    Returns:
        df_merged -- the fully merged dataframe
    """
    # read in the 2022 data
    sales_2022 = pd.read_csv(DATA_DIR + SALES_2022)
        
    # grab relevant columns
    sales_2022_subset = sales_2022[['MK_YR', 'EFFECTIVE_DATE', 'BRAND_FAMILY_DESCR',
                                    'SPECIE_DESCR', 'VARIETY_NAME', 'SLS_LVL_2_ID',
                                    'SUM(ORDER_QTY_TO_DATE)']]
    
    # get the asgrow soybeans
    sales_asgrow = sales_2022_subset[
            sales_2022_subset['BRAND_FAMILY_DESCR'] == 'NATIONAL'].reset_index(drop=True)
    sales_soybeans = sales_asgrow[sales_asgrow['SPECIE_DESCR'] == 'SOYBEAN'].reset_index(drop=True)
    
    # drop those columns
    sales_soybeans = sales_soybeans.drop(columns=['BRAND_FAMILY_DESCR',
                                                  'SPECIE_DESCR'])
    
    # rename the columns
    sales_soybeans = sales_soybeans.rename(columns={'MK_YR': 'year',
                                                    'SLS_LVL_2_ID': 'TEAM_KEY',
                                                    'VARIETY_NAME': 'Variety_Name',
                                                    'EFFECTIVE_DATE': 'date',
                                                    'SUM(ORDER_QTY_TO_DATE)': 'order_Q'})
    
    # remove any variety names with "Empty"
    sales_soybeans = sales_soybeans[
            sales_soybeans['Variety_Name'] != '(Empty)'].reset_index(drop=True)
        
    sales_soybeans = sales_soybeans.merge(abm_Teamkey, on=['TEAM_KEY'], how='left')
    
    sales_soybeans = sales_soybeans.drop(columns=['TEAM_KEY'])
    sales_soybeans['year'] = '2022'
            
    # change the date column to a datetime
    # grab the year, month, and day values from the date field
    sales_dates = sales_soybeans['date'].astype(str).to_frame()
    sales_dates['year'] = sales_dates['date'].str[:4]
    sales_dates['month'] = sales_dates['date'].str[4:6]
    sales_dates['day'] = sales_dates['date'].str[6:8]
    sales_dates = sales_dates.drop(columns=['date'])
    
    sales_dates['EFFECTIVE_DATE'] = pd.to_datetime(sales_dates)
    sales_soybeans['EFFECTIVE_DATE'] = sales_dates['EFFECTIVE_DATE'].copy()
    sales_soybeans = sales_soybeans.drop(columns=['date'])
    
    # only grab orders after certain date
    months=[9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
    year = 2022
    
    # create the total monthly dataframe
    sales_monthly_total = sales_soybeans[
            ['year', 'abm', 'Variety_Name']].drop_duplicates().reset_index(drop=True)
    
    for month in months:
        print("month", month)
        if month > 8:
            date_mask = dt.datetime(year = int(year) - 1, month = month, day = monthrange(int(year) - 1, month)[1])
            
        if month <= 8:
            date_mask = dt.datetime(year = int(year), month = month, day = monthrange(int(year), month)[1])
            
        sales_monthly = sales_soybeans[sales_soybeans['EFFECTIVE_DATE'] <= date_mask].copy().reset_index(drop = True)
        
        sales_monthly = sales_monthly.groupby(
                by=['year','Variety_Name', 'abm'], as_index=False).sum().reset_index(drop = True)
        
        # rename 
        sales_monthly = sales_monthly.rename(columns = {'order_Q': 'order_Q_month_' + str(month)})
            
        # merge with df_monthly_total 
        sales_monthly_total = sales_monthly_total.merge(
                sales_monthly, on=['year', 'Variety_Name', 'abm'], how='left')
    
    sales_monthly_total = sales_monthly_total.fillna(0)
    
    df_w_lag = create_late_lagged_sales(df=sales_monthly_total,
                                        full_df=df,
                                        year=2022)
        
    df_merged = pd.concat([df, df_w_lag])
    
    return df_merged


def preprocess_2020_sale(abm_Teamkey):
    """Preprocesses the 2020 sales data. 
    
    Keyword arguments:
        abm_Teamkey -- the team key to abm converter
    """
    df2020_path = DATA_DIR + SALES_DIR + '2020.csv'
    dfsale_2020 = pd.read_csv(df2020_path)
    dfsale_2020 = dfsale_2020.rename(columns = {'SLS_LVL_2_ID':"TEAM_KEY"})
    dfsale_2020 = dfsale_2020.merge(abm_Teamkey, how = 'left', on = ['TEAM_KEY'])
    dfsale_2020 = dfsale_2020.rename(columns = {'abm':'SLS_LVL_2_ID'})
    
    df2020_path_new = DATA_DIR + SALES_DIR + '2020.csv'
    
    dfsale_2020.to_csv(df2020_path_new, index = False)
    "Finish Proprecssing 2020 sales Data"
    
    
def read_2022_CF_data():
    """Reads in the 2022 y + 1 consensus forecast data
    
    Keyword arguments:
        None
    Returns:
        CF_2022 -- the y + 1 forecast for 2022
    """
    big_cf_file = pd.read_csv(DATA_DIR + BIG_CF_FILE)
    
    # grab the forecast year 2022 piece
    big_cf_file_2022 = big_cf_file[big_cf_file['FORECAST_YEAR'] == 2021].reset_index(drop=True)
    
    # subset out columns
    big_cf_file_2022 = big_cf_file_2022[['FORECAST_YEAR', 'TEAM_KEY', 'ACRONYM_NAME',
                                         'TEAM_Y1_FCST_1']]
    
    # rename the columns
    CF_2022 = big_cf_file_2022.copy().rename(
            columns={'FORECAST_YEAR': 'year',
                     'ACRONYM_NAME': 'Variety_Name'})
        
    CF_2022['year'] = 2022
    
    return CF_2022


def read_2023_CF_data():
    """Reads in the 2023 y+1 consensus forecast data
    
    Keyword arguments:
        None
    Returns:
        CF_2023 -- the y + 1 forecast for 2023
    """
    cf_file = pd.read_excel(DATA_DIR + CF_2022_FILE)
    
    # grab the forecast year 2022 piece
    cf_file_2023 = cf_file[cf_file['FORECAST_YEAR'] == 2022].reset_index(drop=True)
    
    # subset out columns
    cf_file_2023 = cf_file_2023[['FORECAST_YEAR', 'TEAM_KEY', 'ACRONYM_NAME',
                                 'TEAM_Y1_FCST_1']]
    
    # rename the columns
    CF_2023 = cf_file_2023.copy().rename(
            columns={'FORECAST_YEAR': 'year',
                     'ACRONYM_NAME': 'Variety_Name'})
        
    CF_2023['year'] = 2023
    
    return CF_2023


def read_2024_CF_data():
    """Reads in the 2024 y+1 consensus forecast data
    
    Keyword arguments:
        None
    Returns:
        CF_2024 -- the y + 1 forecast for 2024
    """
    cf_file = pd.read_excel(DATA_DIR + CF_2023_FILE)

    cf_file_2024 = cf_file[cf_file['FORECAST_YEAR'] == 2023].reset_index(drop=True)
    
    # subset out columns
    cf_file_2024 = cf_file_2024[['FORECAST_YEAR', 'TEAM_KEY', 'ACRONYM_NAME',
                                 'TEAM_Y1_FCST_1']]
    
    # rename the columns
    CF_2024 = cf_file_2024.copy().rename(
            columns={'FORECAST_YEAR': 'year',
                     'ACRONYM_NAME': 'Variety_Name'})

    CF_2024['year'] = 2024
    
    return CF_2024

def read_abm_teamkey_file():
    """ Reads in and returns the abm and teamkey data as a dataframe.
    
    Keyword arguments:
        None
    Returns:
        abm_Teamkey -- the dataframe  mapping teamkey to abm 
    """
    
    # read in data
    abm_Teamkey_Address = DATA_DIR + ABM_TABLE
    abm_Teamkey = pd.read_csv(abm_Teamkey_Address)
    
    # rename columns 
    abm_Teamkey = abm_Teamkey.rename(columns = {'Old Area ID':'abm', 'New Area ID':'TEAM_KEY'})
    
    # selcted required columns 
    abm_Teamkey = abm_Teamkey[['abm','TEAM_KEY']]
    
    return abm_Teamkey


def read_commodity_corn_soybean():
    """Reads in the commodity data for both corn and soybeans.
    
    Keyword arguments:
        None
    Returns:
        Commodity_Corn -- the soybean commodity data to date
        Commodity_Soybean -- the corn commodity data to date
    """
    # read in soybean and commodity data
    Corn_Address = DATA_DIR + CM_DIR + 'corn_to_01242023.csv'
    Soybean_Address = DATA_DIR + CM_DIR + 'soybean_to_01242023.csv'
    
    Commodity_Corn = pd.read_csv(Corn_Address)
    Commodity_Soybean = pd.read_csv(Soybean_Address)
    
    return Commodity_Corn, Commodity_Soybean


def read_concensus_forecasting():
    """ Reads in and returns the concensus forecasting data as a dataframe.
    
    Keyword arguments:
        None
    Returns:
        CF_2016_2022 -- the dataframe of CF data from 2016 to 2022
    """
    
    # read in data
    CF_2016_2020_Address = DATA_DIR + 'FY16_20_soybean.csv'
    CF_2021_Address = DATA_DIR + 'FY22_01_14_21.csv'
    CF_2016_2020 = pd.read_csv(CF_2016_2020_Address)
    CF_2021 = pd.read_csv(CF_2021_Address, encoding='UTF-8')
    
    # select required columns 
    selected_columns = ['FORECAST_YEAR', 'CROP_DESCR', 'BRAND_GROUP', 'ACRONYM_NAME',
       'TEAM_KEY', 'TEAM_Y1_FCST_1']
    CF_2016_2020 = CF_2016_2020[selected_columns]
    CF_2021 = CF_2021[selected_columns]
    
    # concatenate two dataframe 
    CF_2016_2021 = pd.concat([CF_2016_2020, CF_2021])
    
    # subset crop_descr  = soybean and brand_group = ASGROW
    CF_2016_2021 = CF_2016_2021[(CF_2016_2021['CROP_DESCR'] == 'SOYBEAN') & 
                                ((CF_2016_2021['BRAND_GROUP'] == 'ASGROW') | 
                                 CF_2016_2021['BRAND_GROUP'] == 'NATIONAL')]
    
    # drop unnecessary columns 
    dropped_cols = ['CROP_DESCR','BRAND_GROUP']
    CF_2016_2021 = CF_2016_2021.drop(columns = dropped_cols)
    
    # read in the big file and get hte 2022 stuff
    CF_2022 = read_2022_CF_data()

    CF_2016_2022 = pd.concat([CF_2016_2021, CF_2022])
    
    # rename columns in order to merge
    CF_2016_2022 = CF_2016_2022.rename(columns = {'FORECAST_YEAR':'year',
                                                  'ACRONYM_NAME':'Variety_Name'})
    # convert year to strf
    CF_2016_2022['year'] = CF_2016_2022['year'].astype(int).astype(str)
    
    # drop missing value 
    CF_2016_2022 = CF_2016_2022.dropna(how = 'any')
    
    return CF_2016_2022


def read_CY_CF_data():
    """Reads in the current year consensus data
    
    Keyword Arguments:
        None
    Returns:
        cy_CF -- the current year consensus forecast data
    """
    # read in the file
    big_cf_file = pd.read_csv(DATA_DIR + BIG_CF_FILE)
    
    # subset relevant columns
    big_cf_file_subset = big_cf_file[['FORECAST_YEAR', 'TEAM_KEY', 'ACRONYM_NAME']]
    
    # rename the columns
    cy_CF = big_cf_file_subset.copy().rename(
            columns={'FORECAST_YEAR': 'year',
                     'ACRONYM_NAME': 'Variety_Name'})
    
    return cy_CF


def read_kynetic_data():
    """Reads in the kynetic data.
    
    Keyword arguments:
        None
    Returns:
        kynetic_df_with_abm -- the dataframe of the kynetic data with an ABM 
            feature added
    """
    # read in the kynetic data
    kynetic_df = pd.read_csv(DATA_DIR + KYNETIC_DATA, low_memory=False,
                             sep=',', thousands=',')
    
    # drop unwanted columns
    kynetic_df_abbr = kynetic_df.drop(columns=KYNETIC_COLUMNS_TO_DROP)
    
    # rename the columns to allow for merging with the main dataset
    kynetic_df_renamed = kynetic_df_abbr.rename(columns=KYNETIC_COLUMN_NAMES)
    
    # add the abm feature
    kynetic_df_with_abm = fips_to_abm_by_year(df=kynetic_df_renamed)
    
    # turn the year feature into an object
    kynetic_df_with_abm['year'] = kynetic_df_with_abm['year'].astype('str')
    
    # aggregate by abm, year, and hybrid
    kynetic_df_with_abm = kynetic_df_with_abm.groupby(
            by=['year', 'abm', 'Variety_Name'], as_index=False).mean().reset_index(drop=True)
    
    return kynetic_df_with_abm


def read_performance():
    """ Reads in and returns the performance data as a dataframe.
    
    Keyword arguments:
        None
    Returns:
        Peformance_2011_2022 -- the dataframe of the performance data from 2011 to 2019
    """
    
    # create a list to store all H2H data
    dfs_path = []
    for i in range(2011, 2023):
        print("Read ", str(i), "H2H Data")
        dfi_path = DATA_DIR + H2H_DIR + 'Combined_H2H'+ str(i) + '.csv'
        dfi = pd.read_csv(dfi_path)
        
        # set a year parameter to be the year
        dfi['year'] = i + 1
        dfs_path.append(dfi)
            
        
    # concatenate all H2H data
    Performance_2011_2023 = pd.concat(dfs_path)
    
    # set the 2022 data equal to the 2021 data
    #Performance_2022 = Performance_2011_2021[
    #        Performance_2011_2021['year'] == 2021].copy().reset_index(drop=True)
    #Performance_2022['year'] = 2022
    
    #Performance_2011_2022 = pd.concat([Performance_2011_2021, Performance_2022])
    
    return Performance_2011_2023


def read_sales_filepath(abm_Teamkey):
    """ Reads in and returns the sales data as a dataframe.
    
    Keyword arguments:
        None
    Returns:
        Sale_2012_2020 -- the dataframe of the yealry sales data from 2012 to 2020
        df_clean_sale -- the dataframe of clean sales data with effective date
    """
    # preprocess sales 2020 sales data to get consistent abm data  
    #Preprocess_2020_sale()

    # define a list to store all sales data 
    dfs_path = []
    
    # define a list to store the FULL datasets
    dfs_full = []

    # read in the data by year 
    for year in range(2012, 2021):
        print("Read ", str(year), " Sales Data")
        dfi_path = DATA_DIR + SALES_DIR + str(year) + '.csv'
        dfi = pd.read_csv(dfi_path)
        
        dfi = dfi[dfi['SPECIE_DESCR'] == 'SOYBEAN'].reset_index(drop=True)
        
        # set a year parameter to be the year 
        dfi['year'] = year
        
        # convert the effective date to a datetime format in order to set the mask 
        dfi['EFFECTIVE_DATE'] = pd.to_datetime(dfi['EFFECTIVE_DATE'])
        
        # set 2020 abms to old format
        if year == 2020:
            dfi = dfi.rename(columns = {'SLS_LVL_2_ID':"TEAM_KEY"})
            dfi = dfi.merge(abm_Teamkey, how = 'left', on = ['TEAM_KEY'])
            dfi = dfi.rename(columns = {'abm':'SLS_LVL_2_ID'})
            dfi = dfi.drop(columns=['TEAM_KEY'])
        
        # set the date mask
        date_mask = dt.datetime(year = int(year),
                                month = EFFECTIVE_DATE['month'],
                                day = EFFECTIVE_DATE['day'])
        
        # remove stuff after the effective date
        dfi_masked = dfi[dfi['EFFECTIVE_DATE'] <= date_mask].copy().reset_index(drop=True)
        
        # add modified dataframe to the list 
        dfs_path.append(dfi_masked.copy())
        dfs_full.append(dfi.copy())
      
    # concate all dataframes  
    Sale_2012_2020 = pd.concat(dfs_path).reset_index(drop=True)
    
    Sale_2012_2020_full = pd.concat(dfs_full).reset_index(drop=True)
    
    # rename the columns 
    SALES_COLUMN_NAMES = {'year': 'year', 'SPECIE_DESCR': 'crop',
                          'SLS_LVL_2_ID': 'abm', 'VARIETY_NAME': 'Variety_Name',
                          'NET_SALES_QTY_TO_DATE': 'nets_Q',
                          'ORDER_QTY_TO_DATE': 'order_Q',
                          'RETURN_QTY_TO_DATE': 'return_Q',
                          'REPLANT_QTY_TO_DATE': 'replant_Q'}
    
    # 'eoy' means 'end of year'
    FULL_COLUMN_NAMES = {'year': 'year', 'SPECIE_DESCR': 'crop',
                          'SLS_LVL_2_ID': 'abm', 'VARIETY_NAME': 'Variety_Name',
                          'NET_SALES_QTY_TO_DATE': 'nets_Q_eoy'}
    
    Sale_2012_2020 = Sale_2012_2020.rename(columns = SALES_COLUMN_NAMES)
    Sale_2012_2020_full = Sale_2012_2020_full.rename(columns=FULL_COLUMN_NAMES)
    
    # select the soybean crop 
    Sale_2012_2020 = Sale_2012_2020[
            Sale_2012_2020['BRAND_FAMILY_DESCR'] == 'NATIONAL'].reset_index(drop=True)
    Sale_2012_2020_full = Sale_2012_2020_full[
            Sale_2012_2020_full['BRAND_FAMILY_DESCR'] == 'NATIONAL'].reset_index(drop=True)
    
    # set year as str
    Sale_2012_2020['year'] = Sale_2012_2020['year'].astype(str)
    Sale_2012_2020_full['year'] = Sale_2012_2020_full['year'].astype(str)
    
    
    # drop unnecessary columns for yearly sales
    SALES_COLUMNS_TO_DROP_Yearly = ['BRAND_FAMILY_DESCR', 'EFFECTIVE_DATE',
                                    'DEALER_ACCOUNT_CY_BRAND_FAMILY',
                                    'SHIPPING_STATE_CODE', 'SHIPPING_COUNTY',
                                    'SHIPPING_FIPS_CODE', 'SLS_LVL_1_ID', 'CUST_ID',
                                    'ACCT_ID', 'NET_SHIPPED_QTY_TO_DATE']
    
    Sale_2012_2020 = Sale_2012_2020.drop(columns=SALES_COLUMNS_TO_DROP_Yearly)
    
    # reorder the columns
    Sale_2012_2020 = Sale_2012_2020[['year', 'abm', 'Variety_Name', 
                                    'nets_Q', 'order_Q', 'return_Q','replant_Q']]
    Sale_2012_2020_full = Sale_2012_2020_full[['year', 'abm', 'Variety_Name','nets_Q_eoy']]
    
    Sale_2012_2020 = Sale_2012_2020.groupby(by=['year', 'Variety_Name', 'abm'],
                                            as_index=False).sum()
    Sale_2012_2020_full = Sale_2012_2020_full.groupby(by=['year', 'Variety_Name', 'abm'],
                                                      as_index=False).sum()

    # merge the eoy values
    Sale_2012_2020  = Sale_2012_2020.merge(Sale_2012_2020_full,
                                           on=['year', 'Variety_Name', 'abm'],
                                           how='left')
    
    # add the 2021 data
    Sale_2012_2021 = merge_2021_sales_data_impute_daily(df=Sale_2012_2020,
                                                          abm_Teamkey=abm_Teamkey)
    
    # add the 2022 data
    Sale_2012_2022 = merge_2022_sales_data_impute_daily(df=Sale_2012_2021,
                                                        abm_Teamkey=abm_Teamkey)
    
    Sale_2012_2022 = Sale_2012_2022.fillna(0)
    
    # add in 2023 data , setting nets_Q_eoy to ZERO and use it to create lagged quantities
    Sale_2012_2023 = merge_2023_D1MS(df=Sale_2012_2022,
                                     abm_Teamkey=abm_Teamkey)
    
    # add in 2024 product list
    product_list_24 = pd.read_csv(PROD_LIST_24)
    product_list_24 = product_list_24[['ACRONYM_NAME', 'TEAM_KEY']].merge(
            abm_Teamkey, on=['TEAM_KEY'], how='left').drop(columns=['TEAM_KEY'])
    
    
    product_list_24['year'] = 2024
    product_list_24['nets_Q'] = 0
    product_list_24['order_Q'] = 0
    product_list_24['return_Q'] = 0
    product_list_24['replant_Q'] = 0
    
    product_list_24 = product_list_24.rename(columns={'ACRONYM_NAME': 'Variety_Name'})
        
    # concat
    Sale_2012_2024 = pd.concat([Sale_2012_2023, product_list_24]).reset_index(drop=True)
    
    return Sale_2012_2024



def read_SRP():
    """Reads in the SRP data.
    
    Keyword arguments:
        None
    Returns: 
        SRP_2011_2020 -- the fully concatenated SRP values from 2011 to 2020
    """
    
    # define a list to store all SRP data
    dfs_path = []
    
    # read in the historical data and concatenate each file from 2011 to 2019
    for year in range(2011, 2020):
        print("Read ", str(year), "SRP Data")
        dfi_path = DATA_DIR + HISTORICAL_SRP + str(year) + '_SRP.csv'
        
        dfi = pd.read_csv(dfi_path)
        
        # set a year parameter to be the year 
        dfi['year'] = year
        dfs_path.append(dfi)
    
    # concatenate all dataframes    
    SRP_2011_2019 = pd.concat(dfs_path).reset_index(drop=True)
    
    # select required columns
    SRP_2011_2019 = SRP_2011_2019[['year', 'VARIETY', 'SRP']]
    
    # rename the columns 
    SRP_2011_2019 = SRP_2011_2019.rename(columns={'VARIETY': 'Variety_Name'})
    
    
    # remove any leading or trailing spaces as well as dollar signs
    SRP_2011_2019['SRP'] = SRP_2011_2019['SRP'].str.strip().str.replace('$','')
    
    # remove any null values
    SRP_2011_2019 = SRP_2011_2019[SRP_2011_2019['SRP'] != '-']
    SRP_2011_2019 = SRP_2011_2019.dropna(how = 'any').reset_index(drop = True)
    
    # drop any duplicates
    SRP_2011_2019 = SRP_2011_2019.drop_duplicates().reset_index(drop = True)
    
    # set year as str, SRP as float
    SRP_2011_2019['year'] = SRP_2011_2019['year'].astype(str)
    SRP_2011_2019['SRP'] = SRP_2011_2019['SRP'].astype(float)
    
    # read 2020 SRP data 
    df2020_path = DATA_DIR + HISTORICAL_SRP + '2020_SRP.csv'
    SRP_2020 = pd.read_csv(df2020_path)
    # add a year column
    SRP_2020['year'] = '2020'
    # set price as float 
    SRP_2020['Price'] = SRP_2020['Price'].astype(float)
    # rename columns
    SRP_2020 = SRP_2020.rename(columns={'Product':'Variety_Name', 'Price':'SRP'})
    
    SRP_2011_2020 = pd.concat([SRP_2011_2019, SRP_2020])
    
    # read in the 2021 file
    df_2021_path = DATA_DIR + HISTORICAL_SRP + '21_product_srp.csv'
    SRP_2021 = pd.read_csv(df_2021_path)
    
    SRP_2021['year'] = '2021'
    SRP_2021 = SRP_2021.rename(columns={'Product': 'Variety_Name'})
    
    SRP_2011_2021 = pd.concat([SRP_2011_2020, SRP_2021])
    
    # read in the 2022 file
    df_2022_path = DATA_DIR + HISTORICAL_SRP + '22_product_srp.csv'
    SRP_2022 = pd.read_csv(df_2022_path)
    
    SRP_2022['year'] = '2022'
    SRP_2022 = SRP_2022.rename(columns={'Product Name': 'Variety_Name',
                                        'Srp': 'SRP'})
    SRP_2022 = SRP_2022[['year', 'Variety_Name', 'SRP']]
    
    SRP_2011_2022 = pd.concat([SRP_2011_2021, SRP_2022])
    
    # read in the 2023 file
    df_2023_path = DATA_DIR + HISTORICAL_SRP + '23_product_srp.csv'
    SRP_2023 = pd.read_csv(df_2023_path)
    
    SRP_2023['year'] = '2023'
    SRP_2023 = SRP_2023.rename(columns={'Product Name': 'Variety_Name',
                                        'Srp': 'SRP'})
    SRP_2023 = SRP_2023[['year', 'Variety_Name', 'SRP']]
    
    SRP_2011_2023 = pd.concat([SRP_2011_2022, SRP_2023])
    
    # read in the 24 file
    
    df_2024_path = DATA_DIR + HISTORICAL_SRP + '24_product_srp.csv'
    SRP_2024 = pd.read_csv(df_2024_path)
    
    SRP_2024['year'] = '2024'
    SRP_2024 = SRP_2024.rename(columns={'Product Name': 'Variety_Name',
                                        'Srp': 'SRP'})
    SRP_2024 = SRP_2024[['year', 'Variety_Name', 'SRP']]
    
    SRP_2011_2024 = pd.concat([SRP_2011_2023, SRP_2024])
    
    return SRP_2011_2024


def read_soybean_trait_map():
    """Reads in the SRP data.
    
    Keyword arguments:
        None
    Returns: 
        trait_map -- the trait_map for encoding 
    """
    Address_trait_map = DATA_DIR + 'soybean_trait_map_xf.csv'
    trait_map = pd.read_csv(Address_trait_map)
    
    trait_map = trait_map.fillna(0)
    return trait_map


def read_state_county_fips():
    """ Reads in and returns the performance data as a dataframe.
    
    Keyword arguments:
        None
    Returns:
        State_fips -- the dataframe of state and fips info
        County_fips -- the dataframe of county and fips info 
    """
    State_fips_Address = 'state-geocodes-v2018.xlsx'
    County_fips_Address = 'all-geocodes-v2018.xlsx'
    State_fips = pd.read_excel(State_fips_Address, skiprows = 5)
    County_fips = pd.read_excel(County_fips_Address, skiprows = 4)
    return State_fips, County_fips


def read_weather_filepath():
    """Reads in the Blizzard data and concatenates it into a single dataframe.
       Reads in the county_locations data 
       Reads in the abm_years
    
    Keyword arguments:
        years -- the years we want to read the data for
    Returns:
        Weather_2012_2020 -- the dataframe of all the county level blizzard data
        County_Location - the dataframe of all fips code w.r.t lati and long
        FIPS_abm - the dataframe of all fips w.r.t abm and year 
        
    """
    # define a list to store all weather data 
    dfs_path = []
    
    # read in the data by year
    for i in range(2012, 2025):
        print("Read ", str(i), " Weather Data")
        dfi_path = BLIZZARD_DIR + 'Blizzard_' + str(i) + '.csv'
        dfi = pd.read_csv(dfi_path)
        
        # add the dataframe to the list
        dfs_path.append(dfi)
        
    # concate all dataframes  
    Weather_2012_2020 = pd.concat(dfs_path).reset_index(drop = True)
    
    County_Location_Address = BLIZZARD_DIR + 'county_locations.csv'
    County_Location = pd.read_csv(County_Location_Address)
    
    FIPS_abm_Address = YEARLY_ABM_FIPS_MAP #DATA_DIR + 'abm_years.csv'
    FIPS_abm = pd.read_csv(FIPS_abm_Address)
    FIPS_abm = FIPS_abm[['year', 'fips', 'abm']]
    
    # set the 2023 data to be the same as the 2022
    FIPS_abm_2023 = FIPS_abm[FIPS_abm['year'] == 2022]
    FIPS_abm_2023['year'] = 2023
    FIPS_abm = pd.concat([FIPS_abm, FIPS_abm_2023])
    
    FIPS_abm_2024 = FIPS_abm[FIPS_abm['year'] == 2022]
    FIPS_abm_2024['year'] = 2024
    FIPS_abm = pd.concat([FIPS_abm, FIPS_abm_2024])

    
    # set year as str
    Weather_2012_2020['year'] = Weather_2012_2020['year'].astype(str)
    FIPS_abm['year'] = FIPS_abm['year'].astype(str)
    
    # set fips to int in FIPS_abm
    FIPS_abm['fips'] = FIPS_abm['fips'].astype(int)
    
    return Weather_2012_2020, County_Location, FIPS_abm


def supply_data(df):
    """Reads in historical supply data and sets the supply data for 2023 to be
    the 0.66x the Y1 consensus forecast
    """
    hist_supply = pd.read_csv(HISTORICAL_SUPPLY)
    
    hist_supply['year'] = hist_supply['year'].astype(str)
    
    df_w_supply = df.merge(hist_supply, on=['year', 'abm', 'hybrid'], how='left')
    
    df_w_supply.loc[df_w_supply['avai_supply_region'].isna(), 'avai_supply_region'] = (
           1.33 * df_w_supply.loc[df_w_supply['avai_supply_region'].isna(), 'TEAM_Y1_FCST_1'])
    
    
    return df_w_supply