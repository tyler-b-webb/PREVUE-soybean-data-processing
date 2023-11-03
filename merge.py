#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:22:08 2022

@author: epnzv
"""
import datetime as dt
import pandas as pd

from calendar import monthrange

from aggregation_config import(ABM_FIPS_MAP, CORN_SOY_ACRES, DATA_DIR, ORDER_DATE,
                               ORDER_FRACTION_2021, PRICE_REC, SALES_2021_W_DATE)
from preprocess import (adv_in_trait, adv_outof_trait, adv_overall,
                        yield_aggregation)


def merge_2021_sales_data_w_date(df, abm_Teamkey):
    """Merges the 2021 sales data.
    
    Keyword arguments:
        df -- the dataframe to concat onto
        abm_Teamkey -- the team key to abm converter
    Returns:
        df_merged
    """
    # read in the 2021 sales data
    sales_2021 = pd.read_csv(DATA_DIR + SALES_2021_W_DATE)
    
    # grab relevant columns
    sales_2021_subset = sales_2021[['MK_YR', 'EFFECTIVE_DATE', 'BRAND_FAMILY_DESCR',
                                    'SPECIE_DESCR', 'VARIETY_NAME', 'SLS_LVL_2_ID',
                                    'SUM(ORDER_QTY_TO_DATE)']]
    
    # get the asgrow soybeans
    sales_asgrow = sales_2021_subset[
            sales_2021_subset['BRAND_FAMILY_DESCR'] == 'NATIONAL'].reset_index(drop=True)
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
    sales_soybeans['year'] = '2021'
            
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
    year = 2021
    
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
        
    # create the orders_to_date feature
    this_month = sales_monthly_total['order_Q_month_' + str(ORDER_DATE['month'])].values
    if ORDER_DATE['month'] != 12:
        next_month = sales_monthly_total['order_Q_month_' + str(ORDER_DATE['month'] + 1)].values
    else:
        next_month = sales_monthly_total['order_Q_month_1'].values
    
    this_month_change = next_month - this_month
    sales_monthly_total['orders_to_date'] = this_month + ORDER_FRACTION_2021 * this_month_change
    
    df_merged = df.merge(
            sales_monthly_total, on=['year', 'Variety_Name', 'abm'], how='left')
    
    return df_merged


def merge_advantages(df):
    """Merge yield advantage features, both inside and outside the ABMs.
    
    Keyword arguments:
        df -- the dataframe of the performance data
    Returns:
        adv_df -- the dataframe of the average yield and yield advantages,
            broken down by year, hybrid, trait, and abm. the trait is important
            as we'll use that for imputation
    """
    # aggregate within the abm by trait, out of trait, and overall, mirroring 
    # the df used for the previous years' models
    Performance_yield_adv = df.copy()    
    Performance_in_trait = adv_in_trait(Performance_yield_adv)
    Performance_outof_trait = adv_outof_trait(Performance_yield_adv)
    Performance_overall = adv_overall(Performance_yield_adv)
    yield_overall = yield_aggregation(Performance_yield_adv)
    
    # merge the advantages together
    adv_df = Performance_in_trait.merge(Performance_outof_trait,
                                    on=['year', 'abm', 'trait', 'hybrid'])
    adv_df = adv_df.merge(Performance_overall,
                          on=['year', 'abm', 'trait', 'hybrid'])
    adv_df = adv_df.merge(yield_overall,
                          on=['year', 'abm', 'trait', 'hybrid'])
            
    return adv_df

def merge_cf_with_abm(df_cf, df_abm_key):
    """ Reads in and returns the abm and teamkey data as a dataframe.
    
    Keyword arguments:
        df_cf -- the dataframe of cf data
        df_abm_key -- the dataframe of mapping teamkay to abm 
    Returns:
        CF_abm -- the dataframe of CF data at the abm level  
    """
    # merge CF data with abm 
    CF_abm = df_cf.merge(df_abm_key, how = 'left', on = ['TEAM_KEY'])
    
    # drop missing value
    CF_abm = CF_abm.dropna(how = 'any')

    # select required columns
    CF_abm = CF_abm[['year', 'Variety_Name', 'abm', 'TEAM_Y1_FCST_1']]
    
    return CF_abm


def merge_price_received(df):
    """Merges the price received data by year and abm onto the main dataframe.
    
    Keyword arguments:
        df -- main dataframe
    Returns:
        df_with_price_rec -- the dataframe with the price received data added
    """
    # read in the price received data
    price_rec_raw = pd.read_csv(DATA_DIR + PRICE_REC)
    
    # clean and aggregate the price_rec_raw df
    price_rec = prep_price_rec(df=price_rec_raw)
    
    # split out the two crops
    price_rec_corn = price_rec[price_rec['crop'] == 'CORN'].reset_index(drop=True)
    price_rec_soy = price_rec[price_rec['crop'] == 'SOYBEANS'].reset_index(drop=True)
    
    # drop the crop column and rename the priceRec to include the crop name
    price_rec_soy = price_rec_soy.drop(columns=['crop'])
    price_rec_corn = price_rec_corn.drop(columns=['crop'])
    
    price_rec_soy = price_rec_soy.rename(columns={'priceRec': 'priceRec_soy'})
    price_rec_corn = price_rec_corn.rename(columns={'priceRec': 'priceRec_corn'})
    
    # merge the two
    price_rec_corn_soy = price_rec_soy.merge(price_rec_corn, on=['year', 'abm'])

    # make copies of the data for 2021 and 2022, basically just projecting forward
    df_y_2022 = price_rec_corn_soy[
            price_rec_corn_soy['year'] == 2022].reset_index(drop=True)
    df_y_2023 = df_y_2022.copy()
    df_y_2024 = df_y_2022.copy()
    
    # set the year values to be their appropriate years
    df_y_2023['year'] = 2023
    df_y_2024['year'] = 2024
    
    # concat back in
    price_rec_corn_soy = pd.concat([price_rec_corn_soy, df_y_2023])
    price_rec_corn_soy = pd.concat([price_rec_corn_soy, df_y_2024])

    price_rec_corn_soy['year'] = price_rec_corn_soy['year'].astype(str)

    # merge with the dataframe
    df_with_price_rec = df.merge(price_rec_corn_soy, on=['year', 'abm'], how='left')
    
    # impute the price rec data
    df_with_pr_imputed = impute_price_rec(df=df_with_price_rec)
    df_with_price_rec = df_with_pr_imputed.copy()
        
    return df_with_price_rec


def impute_price_rec(df):
    """Imputes price received data.
    
    Keyword arguments:
        df -- the dataframe including the price received data
    Returns:
        df_imputed -- the dataframe with the the imputed priceRec data
    """
    # grab real valued the year, abm, and priceRec features
    df_features = df[['year', 'priceRec_soy', 'priceRec_corn']]
    df_real = df_features[df_features['priceRec_soy'].isna() != True]
    
    # aggregate the data by year
    df_y = df_real.groupby(by=['year'], as_index=False).mean()
    df_y = df_y.rename(columns={'priceRec_soy': 'priceRec_soy_y',
                                'priceRec_corn': 'priceRec_corn_y'})
        
    df_imputed = df.merge(df_y, on=['year'], how='left')
    
    # set the values
    df_imputed.loc[df_imputed['priceRec_soy'].isna() == True,
                   'priceRec_soy'] = df_imputed.loc[
                           df_imputed['priceRec_soy'].isna() == True, 'priceRec_soy_y']
    df_imputed.loc[df_imputed['priceRec_corn'].isna() == True,
                   'priceRec_corn'] = df_imputed.loc[
                           df_imputed['priceRec_corn'].isna() == True, 'priceRec_corn_y']
    
    # drop the _y values
    df_imputed = df_imputed.drop(columns=['priceRec_corn_y', 'priceRec_soy_y'])
    
    return df_imputed


def prep_price_rec(df):
    """Cleans and aggregates the price received data.
    
    Keyword arguments:
        df -- the raw price received data
    Returns:
        price_rec_prepped -- the cleaned and aggregated price received data
    """
    # grab the state-level values
    df_by_state = df[df['Geo Level'] == 'STATE'].reset_index(drop=True)
    
    # subset out the year, crop, state, and value
    df_by_state = df_by_state[['Year', 'Commodity', 'State', 'Value']]
    
    # drop any non-numeric values
    df_by_state = df_by_state[
            df_by_state['Value'] != ' (NA)'].reset_index(drop=True)
    df_by_state = df_by_state[
            df_by_state['Value'] != ' (D)'].reset_index(drop=True)
    df_by_state = df_by_state[
            df_by_state['Value'] != ' (S)'].reset_index(drop=True)

    
    # make the value column numeric
    df_by_state['Value'] = df_by_state['Value'].astype(float)
    
    # take the average for each state, year, and commodity
    df_averaged = df_by_state.groupby(by=['Year', 'State', 'Commodity'],
                                      as_index=False).mean().reset_index(drop=True)
    
    # create a weighting scheme for each state/abm combination based on acreage
    weighting_map = state_weight()
    
    # merge the weighting map onto the df
    df_with_weight = df_averaged.merge(weighting_map,
                                       on=['Year', 'Commodity', 'State'])
    
    # create a weighted value feature
    df_with_weight['weighted_priceRec'] = (df_with_weight['acreage_ratio'] *
                  df_with_weight['Value'])
    
    # drop the Value, acreage_ratio, and state columns
    df_with_weight = df_with_weight[['Year', 'Commodity', 'abm',
                                     'weighted_priceRec']]
    
    # group by year, crop, and abm and sum
    price_rec_prepped = df_with_weight.groupby(
            by=['Year', 'Commodity', 'abm'],
            as_index=False).sum().reset_index(drop=True)
    
    # rename the columns
    price_rec_prepped = price_rec_prepped.rename(
            columns={
                    'Year': 'year',
                    'Commodity': 'crop',
                    'weighted_priceRec': 'priceRec'}
            )
    
    return price_rec_prepped


def state_weight():
    """Creates a weight for each state/abm combination based on acreage. The
    output should have five columns: year, crop, state, abm, and the ratio of the 
    planting acres in an abm represented by that state divided by the total
    acres in the abm.
    
    Keyword arguments:
        None
    Returns:
        weight_map -- the weighting dataframe
    """
    # read in the acreage data
    acreage_data = pd.read_csv(DATA_DIR + CORN_SOY_ACRES)
    
    # read in the mapping file
    abm_map = pd.read_csv(DATA_DIR + ABM_FIPS_MAP)
    
    # create a unique crd -> abm mapping. some crds belong to multiple abms,
    # but the number of cases is appropriately small (~50 crds out of 350)
    # that the error introduced should small as well
    abm_map = abm_map[['crd', 'abm']].drop_duplicates(
            subset='crd',keep='first').reset_index(drop=True)
    
    # grab the year, crop (commodity), state, state ANSI, ag district code, and 
    # acreage (value) cols
    acreage_data = acreage_data[['Year', 'Commodity', 'State', 'State ANSI',
                                 'Ag District Code', 'Value']]
    
    # create a crd column by concatenating the state ansi and the ag dist code
    acreage_data['crd'] = (acreage_data['State ANSI'].astype(str) + 
                acreage_data['Ag District Code'].astype(str))
    
    # convert the value to float
    acreage_data['crd'] = acreage_data['crd'].astype(float)
    
    # keep only the year, commodity, crd, and value columns
    acreage_data = acreage_data[['Year', 'Commodity', 'State', 'crd', 'Value']]
    
    # merge the abm map onto the dataframe
    acreage_with_abm = acreage_data.merge(abm_map, on=['crd'])
    
    # drop the crd and aggregate by year, crop, abm, and state
    acreage_with_abm = acreage_with_abm.drop(columns=['crd'])
    acreage_agg = acreage_with_abm.groupby(
            by=['Year', 'Commodity', 'State', 'abm'], as_index=False).sum()
    
    # aggregate again by year, crop, and abm
    acreage_abm = acreage_agg.groupby(
            by=['Year', 'Commodity', 'abm'], as_index=False).sum()
    acreage_abm = acreage_abm.rename(columns={'Value': 'total_acres'})
    
    # merge back and create the ratio feature in acreage_agg by dividing the
    # value column by total acreage
    acreage_agg = acreage_agg.merge(acreage_abm, on=['Year', 'Commodity', 'abm'])
    acreage_agg['acreage_ratio'] = (acreage_agg['Value'] / 
               acreage_agg['total_acres'])
    
    # keep the year, crop, state, abm, and ratio columns
    acreage_agg = acreage_agg[['Year', 'Commodity', 'State', 'abm',
                               'acreage_ratio']]
    
    # subset out the 2019 data and copy it for 2020, 21, 22, 23
    for i in range(1, 5):
        acreage_agg_next = acreage_agg[acreage_agg['Year'] == 2019].reset_index(drop=True)
        acreage_agg_next['Year'] = acreage_agg_next['Year'] + i
    
        # concat back in
        acreage_agg = pd.concat([acreage_agg, acreage_agg_next]).reset_index(drop=True)
    
    return acreage_agg
