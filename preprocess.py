#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:07:21 2022

@author: epnzv
"""
import datetime as dt
import pandas as pd
import numpy as np

from calendar import monthrange
from functools import reduce
from pandasql import sqldf

from aggregation_config import (ABM_FIPS_MAP, CF_2022_FILE, DAILY_FRACTIONS, DATA_DIR,
                                E3_EQUAL_XF, MONTHLY_FRACTIONS, ORDER_DATE, ORDER_FRACTION_2021,
                                SALES_2021, SALES_2022, SCM_DATA_DIR, SCM_DATA_FILE, US_STATE_ABBREV,
                                YEARLY_ABM_FIPS_MAP, YIELD_COUNTY_DATA)

def adv_in_trait(df):
    """Aggregates the advantage feature for a given abm within the trait group
    of each product.
    
    Keyword arguments:
        df -- the dataframe of the performance data
    Returns:
        adv_within_trait -- the dataframe with the mean yield advantage
            within the abm and within a product's trait group
    """
    # in_trait: c_trait == o_trait
    in_trait = df[df['c_trait'] == df['o_trait']].reset_index(drop=True)
    
    # selected required columns
    in_trait_abbr = in_trait[['year', 'abm', 'c_hybrid', 'c_trait', 'yield_adv']]
    
    # get the average yield_adv for each abm, year, and product
    adv_within_trait = in_trait_abbr.groupby(by=['year', 'abm', 'c_trait',
                                                 'c_hybrid'], as_index=False).mean()
    adv_within_trait = adv_within_trait.rename(
            columns={'c_trait': 'trait',
                     'c_hybrid': 'hybrid',
                     'yield_adv': 'yield_adv_within_abm_by_trait_brand'})
    
    return adv_within_trait


def adv_outof_trait(df):
    """Aggregates the advantage feature for a given abm outside the trait group
    of each product.
    
    Keyword arguments:
        df -- the dataframe of the performance data
    Returns:
        adv_outside_trait -- the dataframe with the mean yield advantage
            within the abm and outside a product's trait group
    """
    
    # outof_trait: c_trait != o_trait
    outof_trait = df[df['c_trait'] != df['o_trait']].reset_index(drop=True)
    
    # selected required columns
    outof_trait_abbr = outof_trait[['year', 'abm', 'c_hybrid', 'c_trait', 'yield_adv']]
    
    # get the average yield_adv for each abm, year, and product
    adv_outof_trait = outof_trait_abbr.groupby(by=['year', 'abm', 'c_trait', 
                                                   'c_hybrid'], as_index=False).mean()
    adv_outof_trait = adv_outof_trait.rename(
            columns={'c_trait': 'trait',
                     'c_hybrid': 'hybrid',
                     'yield_adv': 'yield_adv_within_abm_outof_trait_brand'})
    
    return adv_outof_trait


def adv_overall(df):
    """Aggregates the advantage feature within a given abm for a product.
    
    Keyword arguments:
        df -- the dataframe  of the performance data
    Returns:
        adv_overall -- the dataframe with the mean yield advantage within the
            abm
    """
    
    # selected required columns
    adv_abbr = df[['year', 'abm', 'c_hybrid', 'c_trait', 'yield_adv']]
    
    # get the average yield_adv for each abm, year, and product
    adv_overall = adv_abbr.groupby(by=['year', 'abm', 'c_trait', 'c_hybrid'],
                                   as_index=False).mean()
    
    adv_overall = adv_overall.rename(
            columns={'c_trait': 'trait',
                     'c_hybrid': 'hybrid',
                     'yield_adv': 'yield_adv_with_abm_brand'})
    
    return adv_overall


def amend_trait_features(df):
    """Standardizes trait features and drops the mapped ones (those are done in PREVUE
    codebase already).
    
    Keyword arguments:
        df -- the dataframe 
        
    Returns:
        df_amended -- the dataframe with amended trait naming
    """
    df_amended = df.copy()
    df_amended.loc[df_amended['trait'] == 'CONV', 'trait'] = 'Conventional'
    
    df_amended = df_amended.drop(columns=['RR', 'SR', 'RR2X', 'XF'])
    
    return df_amended


def clean_commodity(df_soy, df_corn):
    """Cleans the soy and corn commodity data and combines them into a single
    dataframe.
       Aggregates the commodity data by month, crop, etc.
    
    Keyword arguments:
        df_soy -- the dataframe of soy commodity data
        df_corn -- the dataframe of corn commodity data
    
    Returns:
        corn_soybean -- the combined dataframe appropriately aggregated of 
                        the cleaned corn and soy data
    """
    
    # concatenate two dfs
    corn_soybean = pd.concat([df_soy, df_corn])
    
    # only get those rows that have a real valued Price
    corn_soybean= corn_soybean[
            corn_soybean['Price'].notnull()].reset_index(drop=True)
    
    # only get those rows that have a real Contract Date
    corn_soybean = corn_soybean[
            corn_soybean['Contract Date'].notnull()].reset_index(drop=True)
    
    # create datetime format
    corn_soybean['Contract Date'] = pd.to_datetime(corn_soybean['Contract Date'])
    corn_soybean['Update Date'] = pd.to_datetime(corn_soybean['Update Date'])
    
    # create new features for the day, month, and year for the contract and update 
    corn_soybean['cYr'] = corn_soybean['Contract Date'].dt.year
    corn_soybean['cMn'] = corn_soybean['Contract Date'].dt.month
    corn_soybean['cDt'] = corn_soybean['Contract Date'].dt.day
    
    corn_soybean['uYr'] = corn_soybean['Update Date'].dt.year
    corn_soybean['uMn'] = corn_soybean['Update Date'].dt.month
    corn_soybean['uDt'] = corn_soybean['Update Date'].dt.day
    
    # drop unnecesary columns 
    corn_soybean = corn_soybean.drop(columns=['Contract Date', 'Update Date', 'cDt', 'uDt'])
    
    # get the mean price, grouping by the crop and dates
    corn_soybean = corn_soybean.groupby(
            by=['Crop', 'uYr', 'uMn', 'cYr', 'cMn'], as_index=False).mean()
    
    return corn_soybean 


def clean_performance(df):
    """Modify the performance w.r.t year and trait 
    
    Keyword arguments:
        df -- the dataframe of the performance data
    Returns:
        df -- the modified dataframe
    """
    # we are merging the performance data with the sales data by year + 1
    # ie, merging the sales data for 2019 with the yield data for 2018
    # so we add 1 to the year in the adv_features dataframe
    df['year'] = df['year'] + 1
    
    # replace the 'HT3' name with 'XF'
    df['trait'] = df['trait'].replace('HT3', 'XF')
    
    return df


def clean_state_county(State_fips, County_fips, FIPS_abm):
    """ Reads in and returns the performance data as a dataframe.
    
    Keyword arguments:
        State_fips -- the dataframe of state and fips info
        County_fips -- the dataframe of county and fips info
        FIPS_abm - the dataframe of all fips w.r.t abm and year
    Returns:
        State_County_abm -- the dataframe of state_county, fips and abm info
    """
    
    ## clean state files
    us_state_abbrev = US_STATE_ABBREV
    # convert state full name to short abbrev, such as Illinois -> IL 
    state_abbrev = []
    for state in State_fips['Name']:
        if state in us_state_abbrev:
            state_abbrev.append(us_state_abbrev[state])
        else:
            state_abbrev.append("NA")
            
    # create the State variable to store state abbrev
    State_fips['State'] = state_abbrev
    
    # drop missing value
    State_fips = State_fips[State_fips['State'] != "NA"].reset_index()
    
    # select important columns and pad state fips codes: "1" -> "01"
    State_fips = State_fips[['Name','State (FIPS)', 'State']]
    State_fips['State (FIPS)'] = State_fips['State (FIPS)'].astype(str)
    State_fips['State (FIPS)'] = State_fips['State (FIPS)'].str.pad(width=2, side='left', fillchar='0')
    
    
    
    ## clean county files 
    # select important columns
    County_fips = County_fips[[ 'State Code (FIPS)', 'County Code (FIPS)', 'Area Name (including legal/statistical area description)']]
    
    # select information only about county
    Areas = County_fips['Area Name (including legal/statistical area description)']
    County = []
    isCounty = []
    for area in Areas:
        area_list = area.split(" ")
        level = area_list[-1]
        if level == "County":
            isCounty.append("Y")
        else:
            isCounty.append("N")
        area_modified = " ".join(area_list[:-1])
        County.append(area_modified)
    County_fips['County'] = County
    County_fips['isCounty'] = isCounty
    County_fips = County_fips[County_fips['isCounty'] == "Y"]
    
    # pad state and county fips codes and concatenate them together: "1" -> "01 (state) and "1" -> "001" (county) -> "01001"
    County_fips['State Code (FIPS)'] = County_fips['State Code (FIPS)'].astype(str).str.pad(width=2, side='left', fillchar='0')
    County_fips['County Code (FIPS)'] = County_fips['County Code (FIPS)'].astype(str).str.pad(width=3, side='left', fillchar='0')
    County_fips['fips'] = County_fips['State Code (FIPS)'] + County_fips['County Code (FIPS)']
    
    # rename fips columns in order to merge state and county fips dataframe together
    State_fips = State_fips.rename(columns = {'State (FIPS)':'State Code (FIPS)'})
    State_County_fips = State_fips.merge(County_fips, how = "outer", on = ['State Code (FIPS)'])
    State_County_fips = State_County_fips[['Name','State', 'County', 'fips']]
    
    # drop missing values
    State_County_fips = State_County_fips.dropna(how = 'any')

    # get abm level using fips
    FIPS_abm['fips'] = FIPS_abm['fips'].astype(int)
    FIPS_abm['year'] = FIPS_abm['year'].astype(str)
    FIPS_abm['fips'] = FIPS_abm['fips'].astype(str).str.pad(width=5, side='left', fillchar='0')
    State_County_abm = State_County_fips.merge(FIPS_abm, how = "left", on = ['fips'])
    State_County_abm = State_County_abm.dropna(how = 'any')
    
    # rename Columns
    State_County_abm = State_County_abm.rename(columns = {'Name':'state', 'State':'State_abbrev', 'County':'county'})
    State_County_abm = State_County_abm[['state', 'State_abbrev', 'county', 'fips', 'abm']]
    State_County_abm = State_County_abm.drop_duplicates()
    return State_County_abm


def clean_Weather(df_weather, df_county_locations, df_fips):
    """Adds the FIPS code to the Blizzard data, merging on the latitude and 
    longitude.
       Adds the abm feature to the Blizzard data, merging on the fips and year.
       Aggregate the county-level Blizzard data to the abm level. 
    
    Keyword arguments:
        df_weather -- the dataframe of the Blizzard data
        df_county_locations -- the dataframe with fips, latitude, longtitude
        df_fips -- the dataframe with fips, abm, year 
    Returns:
        Weather_w_abm -- the dataframe aggregated to the abm level
    """   
    Weather = df_weather.copy()
    County_Locations = df_county_locations.copy()
    
    # round latitude and longitude 
    Weather['latitude'] = Weather['latitude'].round(2)
    Weather['longitude'] = Weather['longitude'].round(2)
    
    County_Locations['latitude'] = County_Locations['latitude'].round(2)
    County_Locations['longitude'] = County_Locations['longitude'].round(2)
    
    # merge Weather with County_Locations to add the FIPS feature 
    Weather = Weather.merge(County_Locations, on=['latitude', 'longitude'], how='left')
    
    # merge Weather with FIPS_abm to add the abm feature
    Weather = Weather.merge(df_fips, on = ['fips', 'year'], how='left')
    
    # Drop missing value 
    print("Check the fraction of missing value: ", Weather.isna().sum()/Weather.shape[0])
    Weather = Weather.dropna().reset_index(drop = True)
    
    # Drop latitude, longitude, fips 
    dropped_cols = ['latitude', 'longitude', 'fips']
    Weather = Weather.drop(columns = dropped_cols)
    
    # group by the abm, year, and month and take the avg of min/max temperatures
    Weather = Weather.groupby(by=['year', 'month', 'abm'],as_index=False).mean().reset_index(drop=True)
    
    return Weather


def create_commodity_features(df, crop_type):
    """Creates the commodity price features.
    
    Keyword arguments:
        df -- the dataframe of the corn/soy data
        crop_type -- the crop we're creating features for
        
    Returns:
        Commodity_crop -- the dataframe with the newly created commodity features
                          from 1980 to 2022
    """
    # define a list to store all cm dataframes
    dfs_commodity = []
    
    # define the months we will iterate over
    update_months = [1, 2, 3, 4, 5, 6, 7]
    if crop_type == 'soybean':
        contract_months = [7, 8, 9, 11]
    elif crop_type == 'corn':
        contract_months = [7, 9, 12]
        
    # iterate over the contract and update months
    for update_month in update_months:
        for contract_month in contract_months:
            
            # grab the data for the relevant crop 
            df_crop = df[df['Crop'] == crop_type].reset_index(drop = True)
            
            # match the years
            df_commodity = df_crop[
                ((df_crop['uYr'] + 1 == df_crop['cYr']) & 
                 (df_crop['uMn'] == update_month) & 
                 (df_crop['cMn'] == contract_month))].reset_index(drop = True)
            
            # create a timing feature
            df_commodity['timing'] = ('CMprice_' + crop_type + '_' + str(update_month) + 
                      '_' + str(contract_month))
            
            # rename the price feature to include crop name
            df_commodity = df_commodity.rename(columns={'Price': 'CMprice_' + crop_type})
            
            # select relevant columns and rename it 
            df_commodity = df_commodity[['cYr', 'timing', 'CMprice_' + crop_type]]
            df_commodity = df_commodity.rename(columns={'cYr': 'year'})
            dfs_commodity.append(df_commodity)
    
    # concatenate all modified cm dataframes
    Commodity_crop = pd.concat(dfs_commodity)
    
    # reshape the commodity data
    Commodity_crop = Commodity_crop.pivot(index='year', columns='timing', values='CMprice_' + crop_type)
    
    # reset the index to make the year value a column and set all values to be floats
    Commodity_crop = Commodity_crop.reset_index().astype('float64')
    
    Commodity_crop['year'] = Commodity_crop['year'].astype('int32').astype(str)
    
    return Commodity_crop


def create_lagged_features(df_cm):
    """Creates the lagged commodity features in the dataframe that can then be
    merged back into the main df. 
    
    Keyword arguments: 
        df_cm -- the dataframe with the flattened commodity data, broken down by
            by month
    Returns:
        df_corn_soy_lag -- the dataframe with the lagged features added
    """
    # Reset_index 
    df = df_cm.set_index('year')
    
    # create lagged feature
    df_lagged = df.shift(periods = 1)
    
    # rename the lagged columns 
    df_lagged = df_lagged.rename(mapper = lambda x: x + "_1", axis = 1)
    
    # reset index 
    df_lagged.reset_index(inplace = True)
    
    # drop unnecessary columns 
    df_lagged = df_lagged.drop(columns = ['year'])
    
    # concatenate two columns 
    df_corn_soy_lag = pd.concat([df_cm, df_lagged], axis = 1, join = 'inner')
    
    # convert year to str
    df_corn_soy_lag['year'] = df_corn_soy_lag['year'].astype(str)
    return df_corn_soy_lag


def create_imputation_frames(df):
    """Creates dataframes at certain levels used to impute advantage features
    in the main dataframe.
    
    Keyword arguments:
        df -- the dataframe of advantage features
    Returns:
        product_abm_level -- the dataframe with a product/abm level 
        trait_abm_year_level -- the dataframe with a trait/abm/year level 
        abm_year_level -- the dataframe with an abm/year level 
        year_level -- the dataframe with a year level 
    """
    
    # create a product/abm level aggregation, as year-to-year variation is small
    # rename columns for each level 
    ABM_YEAR_LVL_COLS = {
        'yield_adv_within_abm_by_trait_brand': 'yield_adv_within_abm_by_trait_brand_AY',
        'yield_adv_within_abm_outof_trait_brand': 'yield_adv_within_abm_outof_trait_brand_AY',
        'yield_adv_with_abm_brand': 'yield_adv_with_abm_brand_AY',
        'yield': 'yield_AY'}
    
    PRODUCT_ABM_LVL_COLS = {
        'yield_adv_within_abm_by_trait_brand': 'yield_adv_within_abm_by_trait_brand_PA',
        'yield_adv_within_abm_outof_trait_brand': 'yield_adv_within_abm_outof_trait_brand_PA',
        'yield_adv_with_abm_brand': 'yield_adv_with_abm_brand_PA',
        'yield': 'yield_PA'}
    
    TRAIT_ABM_YEAR_LVL_COLS = {
        'yield_adv_within_abm_by_trait_brand': 'yield_adv_within_abm_by_trait_brand_TAY',
        'yield_adv_within_abm_outof_trait_brand': 'yield_adv_within_abm_outof_trait_brand_TAY',
        'yield_adv_with_abm_brand': 'yield_adv_with_abm_brand_TAY',
        'yield': 'yield_TAY'}
    
    YEAR_LVL_COLS = {
        'yield_adv_within_abm_by_trait_brand': 'yield_adv_within_abm_by_trait_brand_Y',
        'yield_adv_within_abm_outof_trait_brand': 'yield_adv_within_abm_outof_trait_brand_Y',
        'yield_adv_with_abm_brand': 'yield_adv_with_abm_brand_Y',
        'yield': 'yield_Y'}
    df['year'] = df['year'].astype(str)
    
    # create a product/abm level aggregation, as year-to-year variation is small
    product_abm_df = df.drop(columns=['year', 'trait'])
    product_abm_level = product_abm_df.groupby(by=['hybrid', 'abm'],
                                               as_index=False).mean()
    product_abm_level = product_abm_level.rename(columns=PRODUCT_ABM_LVL_COLS)
    
    # create a trait/abm/year level aggregation for if product/abm is unavailable
    trait_abm_year_df = df.drop(columns=['hybrid'])
    trait_abm_year_level = trait_abm_year_df.groupby(by=['trait', 'abm', 'year'],
                                                     as_index=False).mean()
    trait_abm_year_level = trait_abm_year_level.rename(columns=TRAIT_ABM_YEAR_LVL_COLS)
    
    # create an abm/year level aggregation for products without a trait value
    abm_year_df = df.drop(columns=['trait', 'hybrid'])
    abm_year_level = abm_year_df.groupby(by=['abm', 'year'],
                                         as_index=False).mean()
    abm_year_level = abm_year_level.rename(columns=ABM_YEAR_LVL_COLS)
    
    # create a year level aggregation so we have values for when we don't have 
    # data for a given abm in a given year
    year_df = df.drop(columns=['trait', 'hybrid', 'abm'])
    year_level = year_df.groupby(by=['year'], as_index=False).mean()
    year_level = year_level.rename(columns=YEAR_LVL_COLS)
    
    return product_abm_level, trait_abm_year_level, abm_year_level, year_level


def create_lagged_sales(df):
    """Creates the "lagged" sales features, namely the sales data for a product
    from the two previous years in a given ABM.
    
    Keyword arguments:
        df -- the dataframe with the cleaned sales data that will be used to 
            create the lagged features
    Returns:
        df_with_lag -- the dataframe with the lagged sales features added
    """
    print('Creating lagged features...')
    
    LAGGED_FEATURES = ['nets_Q_1', 'order_Q_1', 'return_Q_1', 'replant_Q_1',
                   'nets_Q_2', 'order_Q_2', 'return_Q_2']
    
    df = df.rename(columns = {'Variety_Name':'hybrid'})
    df['year'] = df['year'].astype(int)
    sales_all = df.copy()

    # define the selection criteria as strings to use in the sqldf commmand
    # this is principally just for readability
    #current_year_q = "select a.year, a.abm, a.hybrid, a.nets_Q, a.order_Q, a.return_Q, a.replant_Q, "
    #last_year_q = "b.nets_Q as nets_Q_1, b.order_Q as order_Q_1, b.return_Q as return_Q_1, b.replant_Q as replant_Q_1, "
    #two_years_q = "c.nets_Q as nets_Q_2, c.order_Q as order_Q_2, c.return_Q as return_Q_2 from sales_all "
    #join_last_year = "a left join sales_all b on a.hybrid = b.hybrid and a.abm = b.abm and a.year = b.year + 1 "
    #join_two_years = "left join sales_all c on a.hybrid = c.hybrid and a.abm = c.abm and a.year = c.year + 2"
 
    #sales_with_lag = sqldf(current_year_q + last_year_q + two_years_q + 
    #                       join_last_year + join_two_years)
    
    sales_all_lag_1 = sales_all.copy()
    sales_all_lag_2 = sales_all.copy()
    
    # adjust years
    sales_all_lag_1['year'] = sales_all_lag_1['year'] + 1
    sales_all_lag_2['year'] = sales_all_lag_2['year'] + 2
    
    sales_all_lag_1 = sales_all_lag_1[
            ['year', 'hybrid', 'abm', 'nets_Q', 'order_Q', 'return_Q', 'replant_Q']].rename(
            columns={'nets_Q': 'nets_Q_1', 'order_Q': 'order_Q_1',
                     'return_Q': 'return_Q_1', 'replant_Q': 'replant_Q_1'})
    sales_all_lag_2 = sales_all_lag_2[
            ['year', 'hybrid', 'abm', 'nets_Q', 'order_Q', 'return_Q']].rename(
            columns={'nets_Q': 'nets_Q_2', 'order_Q': 'order_Q_2',
                     'return_Q': 'return_Q_2'})
    
    sales_with_lag = sales_all.merge(sales_all_lag_1, on=['year', 'hybrid', 'abm'], how='left')
    sales_with_lag = sales_with_lag.merge(sales_all_lag_2, on=['year', 'hybrid', 'abm'], how='left')
    
    # impute, replacing the NaNs with zeros
    for feature in LAGGED_FEATURES:
        sales_with_lag[feature] = sales_with_lag[feature].fillna(0)
    
    # convert year to str
    sales_with_lag['year'] = sales_with_lag['year'].astype(str)
    
    # rename hybrid columns 
    sales_with_lag = sales_with_lag.rename(columns = {'hybrid':'Variety_Name'})
    
    # grabs data after the cutoff year
    sales_with_lag = sales_with_lag[sales_with_lag['year'] >= '2012'] 

    # drop sales data for this moment 
    #sales_with_lag = sales_with_lag.drop(columns = ['order_Q']).reset_index(drop = True)
    
    return sales_with_lag


def create_late_lagged_sales(df, full_df, year):
    """
    """
       # get the lagged sales data
    df_last_year = full_df[full_df['year'] == str(year - 1)].copy().reset_index(drop=True)
    df_last_year = df_last_year[['abm', 'Variety_Name', 'nets_Q', 'order_Q',
                                 'return_Q', 'replant_Q']]
    df_last_year['year'] = str(year)
    df_last_year = df_last_year.rename(columns={'nets_Q': 'nets_Q_1',
                                                'order_Q': 'order_Q_1',
                                                'return_Q': 'return_Q_1',
                                                'replant_Q': 'replant_Q_1'})
        
    df_two_year = full_df[full_df['year'] == str(year - 2)].copy().reset_index(drop=True)
    df_two_year = df_two_year[['abm', 'Variety_Name', 'nets_Q', 'order_Q',
                               'return_Q', 'replant_Q']]
    df_two_year['year'] = str(year)
    df_two_year = df_two_year.rename(columns={'nets_Q': 'nets_Q_2',
                                              'order_Q': 'order_Q_2',
                                              'return_Q': 'return_Q_2',
                                              'replant_Q': 'replant_Q_2'})

    df_merged = df.merge(df_last_year, 
                         on=['year', 'Variety_Name', 'abm'], 
                         how='left')
    df_merged = df_merged.merge(df_two_year,
                                on=['year', 'Variety_Name', 'abm'],
                                how='left')
    
    return df_merged

  
def create_monthly_sales(Sale_2012_2020_lagged, clean_Sale, abm_Teamkey):
    """ Create the "datemask" to get monthly feature for the netsale data
    
    Keyword arguments:
        Sale_2012_2020 -- the dataframe of the yealry sales data from 2012 to 2020 with lagged features 
        df_clean_sale -- the dataframe of clean sales data with effective date
    Returns:
        Sales_all -- the dataframe of clean sales data with montly and lagged features 
    """
    print('Creating monthly features...')

    clean_Sale = clean_Sale[['EFFECTIVE_DATE', 'year', 'abm', 'Variety_Name', 'order_Q']]
    months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
    
    dfs_monthly_netsales = []
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    for year in years:
        print('year', year)
        df_year = clean_Sale[clean_Sale['year'] == year].reset_index(drop = True)
        
        df_monthly_total = pd.DataFrame()
        
        df_single_year = Sale_2012_2020_lagged[Sale_2012_2020_lagged['year'] == year].copy().reset_index(drop = True)
        
        df_monthly_total['year'] = df_single_year['year'].tolist()
        df_monthly_total['Variety_Name'] = df_single_year['Variety_Name'].tolist()
        df_monthly_total['abm'] = df_single_year['abm'].tolist()
        
        orders_to_date_mask = dt.datetime(year=int(year),
                                          month=ORDER_DATE['month'],
                                          day=ORDER_DATE['day'])
        
        # grab the orders to date
        df_to_date = df_year[
                df_year['EFFECTIVE_DATE'] <= orders_to_date_mask].copy().reset_index(drop=True)
        
        df_to_date = df_to_date.groupby(by=['year', 'Variety_Name', 'abm'],
                                        as_index=False).sum().reset_index(drop=True)
        
        df_to_date = df_to_date.rename(columns={'order_Q': 'orders_to_date'})
        
        for month in months:
            print("month", month)
            if month > 8:
                date_mask = dt.datetime(year = int(year) - 1, month = month, day = monthrange(int(year) - 1, month)[1])
            
            if month <= 8:
                date_mask = dt.datetime(year = int(year), month = month, day = monthrange(int(year), month)[1])
            
            df_monthly = df_year[df_year['EFFECTIVE_DATE'] <= date_mask].copy().reset_index(drop = True)
            
            df_monthly = df_monthly.groupby(by = ['year','Variety_Name', 'abm'], as_index = False).sum().reset_index(drop = True)
            
            # rename 
            df_monthly = df_monthly.rename(columns = {'order_Q': 'order_Q_month_' + str(month)})
            
            # merge with df_montly_total 
            df_monthly_total = df_monthly_total.merge(df_monthly, on = ['year', 'Variety_Name', 'abm'], how = 'left')
        
        # merge the orders to date
        df_monthly_total = df_monthly_total.merge(df_to_date,
                                                  on=['year', 'Variety_Name', 'abm'],
                                                  how='left')
        
        dfs_monthly_netsales.append(df_monthly_total)
     
    Sales_monthly = pd.concat(dfs_monthly_netsales).reset_index(drop = True)    
    Sales_monthly = Sales_monthly.fillna(0)
    
    Sale_all = Sale_2012_2020_lagged.merge(Sales_monthly, on = ['year', 'Variety_Name', 'abm'], how = 'left')
    
    # read in the 2021 data
    Sale_all_2021 = merge_2021_sales_data_impute_monthly(df=Sale_all,
                                                         abm_Teamkey=abm_Teamkey)
    
    # read in the 2022 data
    Sale_all_2022 = merge_2022_SCM_data(df=Sale_all_2021,
                                        abm_Teamkey=abm_Teamkey)
    
    # create lagged sales for the 2023 data
    Sale_all_2022_pred = create_prediction_set(df=Sale_all_2022,
                                               abm_Teamkey=abm_Teamkey)
    
    # fill nas
    Sale_all_2022_pred = Sale_all_2022_pred.fillna(0)
    
    return Sale_all_2022_pred


def create_prediction_set(df, abm_Teamkey):
    """Creates and concatenates the prediction set (2023) onto the total dataframe.
    
    Keyword arguments:
        df -- the dataframe with the training data.
    Returns:
        df_with_pred_set -- the dataframe with the prediction set added
    """
    # initialize a dataframe using the consensus forecast data to build the index
    # we are only grabbing product/abm pairs that have a nonzero Y1_FCST
    CF_2022 = pd.read_excel(DATA_DIR + CF_2022_FILE)
    pred_set_index = CF_2022[
            CF_2022['FORECAST_YEAR'] == 2023].reset_index(drop=True)
    pred_set_index = pred_set_index[
            ['TEAM_KEY', 'ACRONYM_NAME', 'TEAM_Y1_FCST_1']]
    pred_set_index = pred_set_index[
            pred_set_index['TEAM_Y1_FCST_1'] != 0].drop(columns=['TEAM_Y1_FCST_1']).reset_index(drop=True)
        
    # set the new abm using the abm teamkey file
    pred_set_index = pred_set_index.merge(
            abm_Teamkey, on=['TEAM_KEY'], how='left').drop(columns=['TEAM_KEY'])
    
    # rename columns
    pred_set_index = pred_set_index.rename(columns={'ACRONYM_NAME': 'Variety_Name'})
    
    # set the year and set all sale quantities to zero
    pred_set_index['year'] = '2023'
    pred_set_index['nets_Q'] = 0
    pred_set_index['order_Q'] = 0
    pred_set_index['return_Q'] = 0
    pred_set_index['replant_Q'] = 0
    pred_set_index['nets_Q_eoy'] = 0
        
    df_with_pred_set = pd.concat([df, pred_set_index])
    
    return df_with_pred_set


def create_portfolio_weights(df):
    """Uses the dataframe to create the product weights set.
    
    Keyword arguments:
        df -- the full dataset
    Returns:
        None
    """
    weights_matrix = pd.DataFrame(
            columns=['year', 'abm', 'hybrid', 'trait', 'price', 'SRP', 'weight'])
    for year in df['year'].unique():
        if int(year) < 2024:
            single_year = df[df['year'] == str(year)].copy().reset_index(drop=True)
            single_year = single_year[
                    ['year', 'abm', 'age', 'hybrid', 'trait', 'price', 'SRP', 'nets_Q_1', 'TEAM_Y1_FCST_1']]
            
            # set different weights for age one and historical products
            single_year['weight'] = single_year['nets_Q_1'].values
            single_year.loc[
                    single_year['age'] == 1, 'weight'] = single_year.loc[
                            single_year['age'] == 1, 'TEAM_Y1_FCST_1']
            single_year = single_year[
                    ['year', 'abm', 'hybrid', 'trait', 'price', 'SRP', 'weight']]
        else:
            single_year = df[df['year'] == str(year)].copy().reset_index(drop=True)
            single_year = single_year[
                    ['year', 'abm', 'hybrid', 'trait', 'price', 'SRP', 'TEAM_Y1_FCST_1']]
            single_year['weight'] = single_year['TEAM_Y1_FCST_1'].values
            single_year = single_year[
                    ['year', 'abm', 'hybrid', 'trait', 'price', 'SRP', 'weight']]
        if weights_matrix.empty == True:
            weights_matrix = single_year.copy()
        else:
            weights_matrix = pd.concat([weights_matrix, single_year])
            
    weights_matrix['year'] = weights_matrix['year'].astype(str)
    weights_matrix.to_csv('product_weights_w24_new_SRP_impute.csv', index=False)
    
    return weights_matrix


def impute_age_one_lagged(df):
    """
    """
    # grab relevant quantities
    # the logic is to find, for a given ABM, the trend for ORDERS TO DATE
    # and the lagged quantity
    lagged = df[['abm', 'age', 'year', 
                 'nets_Q', 'order_Q', 'replant_Q', 'return_Q']]
    
    lagged = lagged.rename(columns={'order_Q_month_8': 'order_Q'})
    lagged_agg = lagged.groupby(by=['abm', 'year'], as_index=False).sum()
    
    # create ratios
    lagged_agg['nets_Q_ratio'] = lagged_agg['nets_Q'] / lagged_agg['orders_to_date']
    lagged_agg['order_Q_ratio'] = lagged_agg['order_Q'] / lagged_agg['orders_to_date']
    lagged_agg['return_Q_ratio'] = lagged_agg['return_Q'] / lagged_agg['orders_to_date']
    lagged_agg['replant_Q_ratio'] = lagged_agg['replant_Q'] / lagged_agg['orders_to_date']
    
    # make two copies to deprecate the year in and merge
    last_year_agg = lagged_agg.copy()
    two_years_agg = lagged_agg.copy()
    
    last_year_agg['year'] = last_year_agg['year'].astype(int) + 1
    two_years_agg['year'] = two_years_agg['year'].astype(int) + 2
    
    last_year_agg['year'] = last_year_agg['year'].astype(str)
    two_years_agg['year'] = two_years_agg['year'].astype(str)
    
    last_year_agg = last_year_agg[['year', 'abm', 'nets_Q_ratio', 'order_Q_ratio',
                                   'return_Q_ratio', 'replant_Q_ratio']]
    two_years_agg = two_years_agg[['year', 'abm', 'nets_Q_ratio', 'order_Q_ratio',
                                   'return_Q_ratio', 'replant_Q_ratio']]
    
    last_year_agg = last_year_agg.rename(
            columns={'nets_Q_ratio': 'nets_Q_ratio_1',
                     'order_Q_ratio': 'order_Q_ratio_1',
                     'return_Q_ratio': 'return_Q_ratio_1',
                     'replant_Q_ratio': 'replant_Q_ratio_1'})
    two_years_agg = two_years_agg.rename(
            columns={'nets_Q_ratio': 'nets_Q_ratio_2',
                     'order_Q_ratio': 'order_Q_ratio_2',
                     'return_Q_ratio': 'return_Q_ratio_2',
                     'replant_Q_ratio': 'replant_Q_ratio_2'})
    
    # merge
    df_merged = df.merge(last_year_agg, on=['year', 'abm'], how='left')
    df_merged = df_merged.merge(two_years_agg, on=['year', 'abm'], how='left')
    
    # set the return/replant values for age one products based on the CURRENT year
    # order to date
    df_merged.loc[df_merged['age']==1, 'nets_Q_1'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'nets_Q_ratio_1'])
    df_merged.loc[df_merged['age']==1, 'order_Q_1'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'order_Q_ratio_1'])
    df_merged.loc[df_merged['age']==1, 'return_Q_1'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'return_Q_ratio_1'])
    df_merged.loc[df_merged['age']==1, 'replant_Q_1'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'replant_Q_ratio_1'])
    
    df_merged.loc[df_merged['age']==1, 'nets_Q_2'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'nets_Q_ratio_2'])
    df_merged.loc[df_merged['age']==1, 'order_Q_2'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'order_Q_ratio_2'])
    df_merged.loc[df_merged['age']==1, 'return_Q_2'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'return_Q_ratio_2'])
    df_merged.loc[df_merged['age']==1, 'replant_Q_2'] = (
            df_merged.loc[df_merged['age']==1, 'orders_to_date'] * df_merged.loc[df_merged['age']==1, 'replant_Q_ratio_2'])
    
    df_merged.loc[df_merged['age']==2, 'nets_Q_2'] = (
            df_merged.loc[df_merged['age']==2, 'orders_to_date'] * df_merged.loc[df_merged['age']==2, 'nets_Q_ratio_2'])
    df_merged.loc[df_merged['age']==2, 'order_Q_2'] = (
            df_merged.loc[df_merged['age']==2, 'orders_to_date'] * df_merged.loc[df_merged['age']==2, 'order_Q_ratio_2'])
    df_merged.loc[df_merged['age']==2, 'return_Q_2'] = (
            df_merged.loc[df_merged['age']==2, 'orders_to_date'] * df_merged.loc[df_merged['age']==2, 'return_Q_ratio_2'])
    df_merged.loc[df_merged['age']==2, 'replant_Q_2'] = (
            df_merged.loc[df_merged['age']==2, 'orders_to_date'] * df_merged.loc[df_merged['age']==2, 'replant_Q_ratio_2'])
    
    # drop ratios
    df_merged = df_merged.drop(
            columns=['nets_Q_ratio_1', 'order_Q_ratio_1', 'return_Q_ratio_1', 'replant_Q_ratio_1',
                     'nets_Q_ratio_2', 'order_Q_ratio_2', 'return_Q_ratio_2', 'replant_Q_ratio_2'])
    
    return df_merged


def impute_CY_CF(df):
    """
    """
    df_impute = df.copy()
    FITTING_YEARS = [2017, 2018, 2019, 2020, 2021]
    # grab 2017 to 2021 data
    fitting_df = df[df['year'].isin(FITTING_YEARS)]
    
    # set the ratios 
    TEAM_FCST_QTY_9_RATIO = (
            np.sum(fitting_df['TEAM_FCST_QTY_9']) / np.sum(fitting_df['TEAM_Y1_FCST_1']))
    TEAM_FCST_QTY_10_RATIO = (
            np.sum(fitting_df['TEAM_FCST_QTY_10']) / np.sum(fitting_df['TEAM_Y1_FCST_1']))
    TEAM_FCST_QTY_11_RATIO = (
            np.sum(fitting_df['TEAM_FCST_QTY_11']) / np.sum(fitting_df['TEAM_Y1_FCST_1']))
    TEAM_FCST_QTY_12_RATIO = (
            np.sum(fitting_df['TEAM_FCST_QTY_12']) / np.sum(fitting_df['TEAM_Y1_FCST_1']))
    
    # set the values of those years not in the fitting df using the ratio 
    df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_FCST_QTY_9'] = (
            TEAM_FCST_QTY_9_RATIO * df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_Y1_FCST_1'])
    df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_FCST_QTY_10'] = (
            TEAM_FCST_QTY_10_RATIO * df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_Y1_FCST_1'])
    df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_FCST_QTY_11'] = (
            TEAM_FCST_QTY_11_RATIO * df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_Y1_FCST_1'])
    df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_FCST_QTY_12'] = (
            TEAM_FCST_QTY_12_RATIO * df_impute.loc[df['year'].isin(FITTING_YEARS) != True, 'TEAM_Y1_FCST_1'])
    
    return df_impute


def impute_h2h_data(df, product_abm_level, trait_abm_year_level, abm_year_level, year_level):
    """Imputes missing h2h data using the previously aggregated dfs.
    
    Keyword arguments:
        df -- the dataframe with the h2h data merged
        pa_level -- the values aggregated at the product/abm level for use when
            we have missing years
        tay_level -- the values aggregated at the trait/abm/year level for use when 
            we have missing products in a given abm
        ay_level -- the values aggregated at the abm/year level for when we don't have
            trait information for a product in a given abm for a given year
        y_level -- the values aggregated at the year level for when we don't have
            information for an abm for a given year
    Returns:
        df_imputed -- the dataframe with imputed h2h data
    """
    # the idea is to merge the aggregated frames onto the df, then set the actual adv
    # features to be a certain aggregated value based on the logic of what is missing:
    # if we have product data for an abm for some years but not others, use pa_level,
    # if we have the trait data for a given abm/year, tay_level values, etc. it's 
    # an order of preference: pa -> tay -> ay -> y
    H2H_AGG_LEVELS = ['_PA', '_TAY', '_AY', '_Y']
    YIELD_ADV_FEATURES = ['yield_adv_within_abm_by_trait_brand',
                      'yield_adv_within_abm_outof_trait_brand',
                      'yield_adv_with_abm_brand',
                      'yield']
    
    
    product_abm_level = product_abm_level.rename(columns = {'hybrid': 'Variety_Name'})
    df_with_pa = df.merge(product_abm_level, on=['abm', 'Variety_Name'], how='left')
    
    df_with_tay = df_with_pa.merge(trait_abm_year_level, on=['year', 'abm', 'trait'],
                                   how='left')
    
    df_with_ay = df_with_tay.merge(abm_year_level, on=['year', 'abm'], how='left')

    df_with_y = df_with_ay.merge(year_level, on=['year'], how='left')
    
    # the features we'll drop
    agg_features_to_drop = []
    
    # iterate through the levels in the order pa -> tay -> ay -> y
    for level in H2H_AGG_LEVELS:
    # iterate through the features we're going to impute
        for feature in YIELD_ADV_FEATURES:
            # replace missing values with the level of aggregation if it's available
            df_with_y.loc[((df_with_y[feature].isnull() == True) & 
                           (df_with_y[feature + level].isnull() == False)),
        feature] = df_with_y.loc[((df_with_y[feature].isnull() == True) & 
               (df_with_y[feature + level].isnull() == False)), feature + level]
        
            # append the feature + level to drop
            agg_features_to_drop.append(feature + level)
    
    print(agg_features_to_drop)
    df_imputed = df_with_y.drop(columns=agg_features_to_drop)
    
    return df_imputed


def impute_price(df):
    """Imputes the price
    
    Keyword arguments:
        df -- the dataframe
    Returns:
        df_imputed -- the dataframe with the imputed price
    """
    # get a dataframe of the real-valued price + SRP
    price_real = df[df['price'].isnull() == False].reset_index(drop=True)
    price_real = price_real[['year', 'trait', 'abm', 'price', 'SRP']]
    
    # calculate the difference and drop the price and SRP columns
    price_real['diff'] = price_real['price'] - price_real['SRP']
    price_real = price_real.drop(columns=['price', 'SRP'])
    
    # aggregate the difference based on trait/year, year, and trait
    diff_trait_year = price_real.groupby(by=['year', 'trait', 'abm'], as_index=False).mean()
    diff_year = price_real[['year', 'abm', 'diff']].groupby(by=['year', 'abm'], as_index=False).mean()
    diff_trait = price_real[['trait', 'abm', 'diff']].groupby(by=['trait', 'abm'], as_index=False).mean()
    
    #merge these three back into the main df
    diff_trait_year = diff_trait_year.rename(columns={'diff': 'diff_ty'})
    diff_year = diff_year.rename(columns={'diff': 'diff_y'})
    diff_trait = diff_trait.rename(columns={'diff': 'diff_t'})
    
    df_imputed = df.merge(diff_trait_year, on=['year', 'trait', 'abm'], how='left')
    df_imputed = df_imputed.merge(diff_year, on=['year', 'abm'], how='left')
    df_imputed = df_imputed.merge(diff_trait, on=['trait', 'abm'], how='left')
    
    # set a new set of prices based on the difference and the SRP
    df_imputed['price_ty'] = df_imputed['SRP'] + df_imputed['diff_ty']
    df_imputed['price_y'] = df_imputed['SRP'] + df_imputed['diff_y']
    df_imputed['price_t'] = df_imputed['SRP'] + df_imputed['diff_t']
    
    df_imputed = df_imputed.drop(columns=['diff_ty', 'diff_y', 'diff_t'])
    
    # set missing values to be the aggregated values. trait/year if it is available,
    # just by year if not
    df_imputed.loc[((df_imputed['price'].isnull() == True) & 
                    (df_imputed['price_ty'].isnull() == False)), 
    'price'] = df_imputed.loc[((df_imputed['price'].isnull() == True) & 
         (df_imputed['price_ty'].isnull() == False)), 'price_ty']
    
    df_imputed.loc[((df_imputed['price'].isnull() == True) & 
                    (df_imputed['price_y'].isnull() == False)), 
    'price'] = df_imputed.loc[((df_imputed['price'].isnull() == True) & 
         (df_imputed['price_y'].isnull() == False)), 'price_y']
    
    df_imputed.loc[((df_imputed['price'].isnull() == True) & 
                    (df_imputed['price_t'].isnull() == False)),
    'price'] = df_imputed.loc[((df_imputed['price'].isnull() == True) & 
         (df_imputed['price_t'].isnull() == False)), 'price_t']
   
    # if i genuinely can't fill any of these with something NAN, fill with SRP
    df_imputed.loc[df_imputed['price'].isnull() == True, 
                   'price'] = df_imputed.loc[df_imputed['price'].isnull() == True, 'SRP']
    
    # fill places where price > SRP with SRP
    df_imputed['price'] = np.where(
            df_imputed['price'] < df_imputed['SRP'],
            df_imputed['price'],
            df_imputed['SRP'])
    
    # drop the aggregated values, leaving just the price column with the newly
    # imputed values
    df_imputed = df_imputed.drop(columns=['price_ty', 'price_y', 'price_t'])

    return df_imputed


def impute_SRP(df):
    """Imputes missing SRP values.
    
    Keyword arguments:
        df -- the dataframe with the sales data
    Returns:
        df_imputed -- the dataframe with imputed SRP values
    """
    # create a dataframe of the real-valued SRP values and grab SRP, trait, and year
    SRP_real = df[df['SRP'].isnull() == False].reset_index(drop=True)
    SRP_last_year = SRP_real[['year', 'Variety_Name', 'SRP']].drop_duplicates().reset_index(drop=True)
    SRP_real = SRP_real[['year', 'trait', 'SRP']]
    
    # get the last year SRPs
    SRP_last_year['year'] = SRP_last_year['year'].astype(int) + 1
    SRP_last_year['year'] = SRP_last_year['year'].astype(str)
    SRP_last_year = SRP_last_year.rename(columns={'SRP': 'SRP_ly'})
    
    # aggregate the SRP_real based on trait/year, year, and trait
    SRP_trait_year = SRP_real.groupby(by=['year', 'trait'], as_index=False).mean()
    SRP_year = SRP_real[['year', 'SRP']].groupby(by=['year'], as_index=False).mean()
    SRP_trait = SRP_real[['trait', 'SRP']].groupby(by=['trait'], as_index=False).mean()
    
    # merge these three back into the main df
    SRP_trait_year = SRP_trait_year.rename(columns={'SRP': 'SRP_ty'})
    SRP_year = SRP_year.rename(columns={'SRP': 'SRP_y'})
    SRP_trait = SRP_trait.rename(columns={'SRP': 'SRP_t'})
    
    df_imputed = df.merge(SRP_last_year, on=['year', 'Variety_Name'], how='left')
    df_imputed = df_imputed.merge(SRP_trait_year, on=['year', 'trait'], how='left')
    df_imputed = df_imputed.merge(SRP_year, on=['year'], how='left')
    df_imputed = df_imputed.merge(SRP_trait, on=['trait'], how='left')
    
    # set missing values to be the aggregated values. trait/year if it is available,
    # just by year if not
    df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
                    (df_imputed['SRP_ly'].isnull() == False)), 
    'SRP'] = df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
         (df_imputed['SRP_ly'].isnull() == False)), 'SRP_ly']
    
    df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
                    (df_imputed['SRP_ty'].isnull() == False)), 
    'SRP'] = df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
         (df_imputed['SRP_ty'].isnull() == False)), 'SRP_ty']
    
    df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
                    (df_imputed['SRP_y'].isnull() == False)), 
    'SRP'] = df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
         (df_imputed['SRP_y'].isnull() == False)), 'SRP_y']
    
    df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
                    (df_imputed['SRP_t'].isnull() == False)),
    'SRP'] = df_imputed.loc[((df_imputed['SRP'].isnull() == True) & 
         (df_imputed['SRP_t'].isnull() == False)), 'SRP_t']
    
    # drop the aggregated values, leaving just the SRP column with the newly
    # imputed values
    df_imputed = df_imputed.drop(columns=['SRP_ly', 'SRP_ty', 'SRP_y', 'SRP_t'])

    return df_imputed


def flatten_monthly_weather(df_weather):
    """Flattens the commodity dataframe in such a way that each month in a year
    gets a column. The number of rows will be the number of years and abms, and 
    the number of columns the number of months * the number of weather features
    + 2 for the year and abm. The data inside is the weather data itself.
    
    Keyword arguments:
        df_weather -- the unflattened dataframe with the weather data
    Returns:
        weather_flattened -- the data flattened as detailed above
    """
    
    # define a list to store all flattened weather data 
    dfs_flattened = []
    df = df_weather.copy()
    
    # get the month range 
    weather_month = df['month'].unique().tolist()
    
    # get weather features 
    weather_features = ['precipitation', 'total_solar_radiation','minimum_temperature', 'maximum_temperature']
    
    for month in weather_month:
        df_month = df[df['month'] == month].reset_index(drop = True)
        
        for feature in weather_features:
            df_month = df_month.rename(columns = {feature: feature + "_" + str(month)})
        
        # drop unnecessary columns
        df_month = df_month.drop(columns=['month']) 
        dfs_flattened.append(df_month)
    
    # merge all dfs together 
    weather_flattened = reduce(lambda df1,df2: pd.merge(df1,df2,how = 'left', on=['year', 'abm']), dfs_flattened)
    return weather_flattened


def get_RM(df):
    """
    """
    hybrids = df['Variety_Name']
    
    digits = hybrids.str.findall('[0-9]+')
    
    for i in range(len(digits)):
        if len(digits[i]) > 0:
            digits[i] = digits[i][0]
        else:
            digits[i] = '00'
    
    # turn the digits series into a df
    digits = digits.to_frame(name='digits')
    
    # set the RM to be 0 and then 
    digits['RM'] = 0
    digits['Variety_Name'] = hybrids.values
    
    digits = digits.drop_duplicates().reset_index(drop=True)
    for i in range(len(digits)):
        if int(digits.loc[i, 'digits'][:2]) <= 0:
            digits.loc[i, 'RM'] = -0.1
        elif (int(digits.loc[i, 'digits'][:2]) > 0  and int(digits.loc[i, 'digits'][:2]) < 5):
            digits.loc[i, 'RM'] = 0
        elif (int(digits.loc[i, 'digits'][:2]) >= 5 and int(digits.loc[i, 'digits'][:2]) < 10):
            digits.loc[i, 'RM'] = 0.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 10 and int(digits.loc[i, 'digits'][:2]) < 15):
            digits.loc[i, 'RM'] = 1.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 15 and int(digits.loc[i, 'digits'][:2]) < 20):
            digits.loc[i, 'RM'] = 1.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 20 and int(digits.loc[i, 'digits'][:2]) < 25):
            digits.loc[i, 'RM'] = 2.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 25 and int(digits.loc[i, 'digits'][:2]) < 30):
            digits.loc[i, 'RM'] = 2.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 30 and int(digits.loc[i, 'digits'][:2]) < 35):
            digits.loc[i, 'RM'] = 3.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 35 and int(digits.loc[i, 'digits'][:2]) < 40):
            digits.loc[i, 'RM'] = 3.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 40 and int(digits.loc[i, 'digits'][:2]) < 45):
            digits.loc[i, 'RM'] = 4.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 45 and int(digits.loc[i, 'digits'][:2]) < 50):
            digits.loc[i, 'RM'] = 4.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 50 and int(digits.loc[i, 'digits'][:2]) < 55):
            digits.loc[i, 'RM'] = 5.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 55 and int(digits.loc[i, 'digits'][:2]) < 60):
            digits.loc[i, 'RM'] = 5.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 60 and int(digits.loc[i, 'digits'][:2]) < 65):
            digits.loc[i, 'RM'] = 6.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 65 and int(digits.loc[i, 'digits'][:2]) < 70):
            digits.loc[i, 'RM'] = 6.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 70 and int(digits.loc[i, 'digits'][:2]) < 75):
            digits.loc[i, 'RM'] = 7.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 75 and int(digits.loc[i, 'digits'][:2]) < 80):
            digits.loc[i, 'RM'] = 7.5
        elif (int(digits.loc[i, 'digits'][:2]) >= 80 and int(digits.loc[i, 'digits'][:2]) < 85):
            digits.loc[i, 'RM'] = 8.0
        elif (int(digits.loc[i, 'digits'][:2]) >= 85 and int(digits.loc[i, 'digits'][:2]) < 90):
            digits.loc[i, 'RM'] = 8.5
            
    df_w_RM = df.merge(digits.drop(columns=['digits']),
                       on=['Variety_Name'],
                       how='left')
    
    return df_w_RM


def merge_2021_sales_data_impute_daily(df, abm_Teamkey):
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
                                    'Returns', 'Haulbacks', 'Replants', 'Shipped']]
    
    # rename the columns
    sales_2021_subset = sales_2021_subset.rename(columns={'Team': 'TEAM_KEY',
                                                          'VARIETY': 'Variety_Name',
                                                          'CY Net Sales': 'nets_Q',
                                                          'Returns': 'return_Q_ret',
                                                          'Haulbacks': 'return_Q_haul',
                                                          'Replants': 'replant_Q',
                                                          'Shipped': 'order_Q'
                                                          })
    
    # remove any variety names with "Empty"
    sales_2021_subset = sales_2021_subset[
            sales_2021_subset['Variety_Name'] != '(Empty)'].reset_index(drop=True)
        
    sales_2021_subset = sales_2021_subset.merge(abm_Teamkey, on=['TEAM_KEY'],
                                                how='left')
    
    sales_2021_subset = sales_2021_subset.drop(columns=['TEAM_KEY'])
    
    # change order data to float
    sales_2021_subset['order_Q'] = sales_2021_subset['order_Q'].str.replace(',','')
    sales_2021_subset['order_Q'] = sales_2021_subset['order_Q'].astype('float64')
    
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
    
    # read in the monthly historical fractions
    daily_fractions = pd.read_csv(DAILY_FRACTIONS)
    
    # grab the day from the daily fractions
    daily_fractions_order_date = daily_fractions[
            (daily_fractions['month'] == ORDER_DATE['month']) &
            (daily_fractions['day'] == ORDER_DATE['day'])].reset_index(drop=True)
    
    daily_fractions_order_date = daily_fractions_order_date.drop(columns=['month', 'day'])
    
    # merge the monthly fractions
    sales_2021_to_date = sales_2021_agg.merge(daily_fractions_order_date,
                                              on=['abm'],
                                              how='left')
    
    # drop NAs
    sales_2021_to_date = sales_2021_to_date.dropna().reset_index(drop=True)
            
    quantities = ['nets', 'order', 'return', 'replant']
    
    sales_2021_to_date['nets_Q_eoy'] = sales_2021_to_date['nets_Q'].values
    
    for qty in quantities:
        sales_2021_to_date[qty + '_Q'] = sales_2021_to_date[qty + '_Q'] * sales_2021_to_date[qty + '_fraction']
        sales_2021_to_date = sales_2021_to_date.drop(columns=[qty + '_fraction'])
        
    sales_2021_to_date['year'] = 2021
    sales_2021_to_date['year'] = sales_2021_to_date['year'].astype(str)    
    
    # concatenate with the main dataframe
    df_merged = pd.concat([df, sales_2021_to_date])
    
    return df_merged


def merge_2021_sales_data_impute_monthly(df, abm_Teamkey):
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
                                                          'Replants': 'replant_Q',
                                                          'Orders': 'order_Q'})
    
    # remove any variety names with "Empty"
    sales_2021_subset = sales_2021_subset[
            sales_2021_subset['Variety_Name'] != '(Empty)'].reset_index(drop=True)
        
    sales_2021_subset = sales_2021_subset.merge(abm_Teamkey, on=['TEAM_KEY'],
                                                how='left')
    
    sales_2021_subset = sales_2021_subset.drop(columns=['TEAM_KEY'])
        
    # change order data to float
    sales_2021_subset['order_Q'] = sales_2021_subset['order_Q'].str.replace(',','')
    sales_2021_subset['order_Q'] = sales_2021_subset['order_Q'].astype('float64')
    
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
    
    # get the end of year nets_Q
    sales_2021_agg_nets_Q = sales_2021_agg[['Variety_Name', 'abm', 'nets_Q']].copy()
    sales_2021_agg_nets_Q = sales_2021_agg_nets_Q.rename(
            columns={'nets_Q': 'nets_Q_eoy'})
    
    # read in the monthly historical fractions
    monthly_fractions = pd.read_csv(MONTHLY_FRACTIONS)
    
    # merge the monthly fractions
    sales_2021_monthly = sales_2021_agg.merge(monthly_fractions,
                                              on=['abm'],
                                              how='left')
    
    # drop NAs
    sales_2021_monthly = sales_2021_monthly.dropna().reset_index(drop=True)
        
    # create the monthly order features
    months = [9, 10, 11, 12, 1,2, 3, 4, 5, 6, 7]
    quantities = ['order_Q', 'return_Q', 'replant_Q', 'nets_Q']
    
    # impute the quantities using the monthly fractions
    for qty in quantities:
        for i in months:
            sales_2021_monthly[qty + '_month_' + str(i)] = (
                    sales_2021_monthly[qty] * sales_2021_monthly['frac_' + str(i)])
        sales_2021_monthly[qty + '_month_8'] = sales_2021_monthly[qty].values
        if ORDER_DATE['month'] != 1:
            this_month = sales_2021_monthly[qty + '_month_' + str(ORDER_DATE['month'] - 1)].values
        else:
            this_month = sales_2021_monthly[qty + '_month_12'].values
        
        next_month = sales_2021_monthly[qty + '_month_' + str(ORDER_DATE['month'])].values
        this_month_change = next_month - this_month
        
        sales_2021_monthly[qty] = this_month + ORDER_FRACTION_2021 * this_month_change
        
        # drop monthly quantities
        sales_2021_monthly = sales_2021_monthly.drop(
                columns=[qty + '_month_9', qty + '_month_10', qty + '_month_11',
                         qty + '_month_12', qty + '_month_1', qty + '_month_2',
                         qty + '_month_3', qty + '_month_4', qty + '_month_5',
                         qty + '_month_6', qty + '_month_7', qty + '_month_8'])
    
    sales_2021_monthly['year'] = '2021'
                
    # drop the fractional quantities
    sales_2021_monthly = sales_2021_monthly.drop(
            columns=['frac_9', 'frac_10', 'frac_11', 'frac_12', 'frac_1',
                     'frac_2', 'frac_3', 'frac_4', 'frac_5', 'frac_6', 'frac_7'])
    
    # merge the eoy sales
    sales_2021_monthly_w_eoy = sales_2021_monthly.merge(sales_2021_agg_nets_Q, 
                                                        on=['Variety_Name', 'abm'],
                                                        how='left')
        
    # concatenate with the main dataframe
    df_merged = pd.concat([df, sales_2021_monthly_w_eoy])
    
    return df_merged


def merge_2022_sales_data_impute_daily(df, abm_Teamkey):
    """Merges the 2022 sales data.
    
    Keyword arguments:
        df -- the dataframe to concat onto
    Returns:
        df_merged
    """
    # read in the 2021 sales data
    sales_2022 = pd.read_csv(DATA_DIR + SALES_2022)
    
    # grab relevant columns
    sales_2022_subset = sales_2022[['Team', 'VARIETY', 'CY Net Sales',
                                    'Returns', 'Haulbacks', 'Replants', 'Shipped']]
    
    # rename the columns
    sales_2022_subset = sales_2022_subset.rename(columns={'Team': 'TEAM_KEY',
                                                          'VARIETY': 'Variety_Name',
                                                          'CY Net Sales': 'nets_Q',
                                                          'Returns': 'return_Q_ret',
                                                          'Haulbacks': 'return_Q_haul',
                                                          'Replants': 'replant_Q',
                                                          'Shipped': 'order_Q'})
    
    # remove any variety names with "Empty"
    sales_2022_subset = sales_2022_subset[
            sales_2022_subset['Variety_Name'] != '(Empty)'].reset_index(drop=True)
        
    sales_2022_subset = sales_2022_subset.merge(abm_Teamkey, on=['TEAM_KEY'],
                                                how='left')
    
    sales_2022_subset = sales_2022_subset.drop(columns=['TEAM_KEY'])
    
    # change order data to float
    sales_2022_subset['order_Q'] = sales_2022_subset['order_Q'].str.replace(',','')
    sales_2022_subset['order_Q'] = sales_2022_subset['order_Q'].astype('float64')
    
    # do the same for the returns/haulbacks and set return_Q to be the sum of the quantities
    sales_2022_subset['return_Q_ret'] = sales_2022_subset['return_Q_ret'].str.replace(',','')
    sales_2022_subset['return_Q_ret'] = sales_2022_subset['return_Q_ret'].astype('float64')
    sales_2022_subset['return_Q_haul'] = sales_2022_subset['return_Q_haul'].str.replace(',','')
    sales_2022_subset['return_Q_haul'] = sales_2022_subset['return_Q_haul'].astype('float64')
    
    # do the same for nets Q and replants
    sales_2022_subset['nets_Q'] = sales_2022_subset['nets_Q'].str.replace(',', '')
    sales_2022_subset['nets_Q'] = sales_2022_subset['nets_Q'].astype('float64')
    sales_2022_subset['replant_Q'] = sales_2022_subset['replant_Q'].str.replace(',', '')
    sales_2022_subset['replant_Q'] = sales_2022_subset['replant_Q'].astype('float64')

    sales_2022_subset['return_Q'] = (
            sales_2022_subset['return_Q_haul'] + sales_2022_subset['return_Q_ret'])
    
    sales_2022_subset = sales_2022_subset.drop(columns=['return_Q_haul',
                                                        'return_Q_ret'])

    # group by product and abm
    sales_2022_agg = sales_2022_subset.groupby(
            by=['Variety_Name', 'abm'], as_index=False).sum()
    
    # read in the monthly historical fractions
    daily_fractions = pd.read_csv(DAILY_FRACTIONS)
    
    # grab the day from the daily fractions
    daily_fractions_order_date = daily_fractions[
            (daily_fractions['month'] == ORDER_DATE['month']) &
            (daily_fractions['day'] == ORDER_DATE['day'])].reset_index(drop=True)
    
    daily_fractions_order_date = daily_fractions_order_date.drop(columns=['month', 'day'])
    
    # merge the monthly fractions
    sales_2022_to_date = sales_2022_agg.merge(daily_fractions_order_date,
                                              on=['abm'],
                                              how='left')
    
    # drop NAs
    sales_2022_to_date = sales_2022_to_date.dropna().reset_index(drop=True)
    
    sales_2022_to_date['nets_Q_eoy'] = sales_2022_to_date['nets_Q'].values

    
    quantities = ['nets', 'order', 'return', 'replant']
    
    for qty in quantities:
        sales_2022_to_date[qty + '_Q'] = sales_2022_to_date[qty + '_Q'] * sales_2022_to_date[qty + '_fraction']
        sales_2022_to_date = sales_2022_to_date.drop(columns=[qty + '_fraction'])

    sales_2022_to_date['year'] = 2022
    sales_2022_to_date['year'] = sales_2022_to_date['year'].astype(str)    
    
    # concatenate with the main dataframe
    df_merged = pd.concat([df, sales_2022_to_date])
    
    return df_merged


def merge_2022_SCM_data(df, abm_Teamkey):
    """
    """
    # read in the SCM data
    SCM_data = pd.read_csv(DATA_DIR + SCM_DATA_DIR + SCM_DATA_FILE)
    
    SCM_data_subset = SCM_data[['Team', 'VARIETY', 'Dealer/Gross Orders', 
                                'Returns', 'Replants', 'Haulbacks', 'CY Net Sales']]
    
    SCM_data_subset = SCM_data_subset.rename(
            columns={'Team': 'TEAM_KEY',
                     'VARIETY': 'Variety_Name',
                     'Dealer/Gross Orders': 'order_Q',
                     'Returns': 'return_Q',
                     'Haulbacks': 'haulback_Q',
                     'Replants': 'replant_Q',
                     'CY Net Sales': 'nets_Q'})
        
    # remove commas and convert values to int
    quantities = ['order_Q', 'return_Q', 'haulback_Q', 'replant_Q', 'nets_Q']
    for qty in quantities:
        print(qty)
        
        # if the qty is stored as a string, remove commas
        if pd.api.types.is_string_dtype(SCM_data_subset[qty]) == True:   
            SCM_data_subset[qty] = SCM_data_subset[qty].str.strip().str.replace(',','')
        SCM_data_subset = SCM_data_subset.fillna(0)
        SCM_data_subset[qty] = SCM_data_subset[qty].astype(int)
        
    # add the haulbacks to the return quantity
    SCM_data_subset['return_Q'] = SCM_data_subset['return_Q'] + SCM_data_subset['haulback_Q']
    SCM_data_subset = SCM_data_subset.drop(columns=['haulback_Q'])
    
    # drop missing variety names
    SCM_data_subset = SCM_data_subset[
            SCM_data_subset['Variety_Name'] != '(Empty)'].reset_index(drop=True)
        
    SCM_data_subset = SCM_data_subset.merge(abm_Teamkey, on=['TEAM_KEY'], how='left')
    
    SCM_data_subset = SCM_data_subset.drop(columns=['TEAM_KEY'])
    SCM_data_subset['year'] = '2022'
    
    # aggregate
    SCM_data_subset_agg = SCM_data_subset.groupby(
            by=['year', 'Variety_Name', 'abm'], as_index=False).sum().reset_index(drop=True)

    SCM_data_subset_agg = SCM_data_subset_agg.fillna(0)
        
    df_merged = pd.concat([df, SCM_data_subset_agg])
    
    return df_merged


def merge_2023_D1MS(df, abm_Teamkey):
    """
    """
    
    # read in the file
    sales_23 = pd.read_csv(DATA_DIR + 'D1_MS_23_product_location_022823.csv')
    
    # fill nas with 0
    sales_23 = sales_23.fillna(0)
    
    sales_23 = sales_23.rename(columns={'MK_YR': 'year', 'VARIETY_NAME': 'Variety_Name',
                                'SLS_LVL_2_ID': 'TEAM_KEY', 
                                'SUM(NET_SALES_QTY_TO_DATE)': 'nets_Q',
                                'SUM(ORDER_QTY_TO_DATE)': 'order_Q',
                                'SUM(RETURN_QTY_TO_DATE)': 'return_Q',
                                'SUM(REPLANT_QTY_TO_DATE)': 'replant_Q'})
    
    # select national brand corn
    sales_23 = sales_23[sales_23['BRAND_FAMILY_DESCR'] == 'NATIONAL'].reset_index(drop=True)
    sales_23 = sales_23[sales_23['SPECIE_DESCR'] == 'SOYBEAN'].reset_index(drop=True)
    
    sales_23_subset = sales_23[
            ['year', 'Variety_Name', 'TEAM_KEY', 'EFFECTIVE_DATE', 'nets_Q',
             'order_Q', 'return_Q', 'replant_Q']]
    
    # drop all nas
    sales_23_subset = sales_23_subset.dropna().reset_index(drop=True)
    
    # re-adjust to old abm names
    sales_23_subset = sales_23_subset.merge(abm_Teamkey, on=['TEAM_KEY'], how='left')
    sales_23_subset = sales_23_subset.drop(columns=['TEAM_KEY'])
    
    # remove the 'RIB' string from the hybrid name
    sales_23_subset['Variety_Name'] = sales_23_subset['Variety_Name'].str.replace('RIB', '')

    # change the effective date string to make the datetime readable
    sales_23_dates = sales_23_subset['EFFECTIVE_DATE'].astype(str).to_frame()
    
    sales_23_dates['year'] = sales_23_dates['EFFECTIVE_DATE'].str[:4]
    sales_23_dates['month'] = sales_23_dates['EFFECTIVE_DATE'].str[4:6]
    sales_23_dates['day'] = sales_23_dates['EFFECTIVE_DATE'].str[6:8]
    
    sales_23_dates = sales_23_dates.drop(columns=['EFFECTIVE_DATE'])
    
    # create a datetime object out of the effective date
    sales_23_subset['EFFECTIVE_DATE'] = pd.to_datetime(sales_23_dates)
    
    # remove M string from the year
    sales_23_subset['year'] = sales_23_subset['year'].str.replace('M', '')
    
    sales_23_no_date = sales_23_subset.drop(columns=['EFFECTIVE_DATE']).groupby(
            by=['year', 'Variety_Name', 'abm'], as_index=False).sum()
    
    # set the net sales quantity (use forecasted values here?)
    sales_23_no_date['nets_Q_eoy'] = 0
    
    df_w_23 = pd.concat([df, sales_23_no_date])
    
    return df_w_23


def Performance_with_yield_adv(df_performance):
    """Creates yield advantage features
    
    Keyword arguments:
        df_performance -- the dataframe of the performance data
    Returns:
        Performance_abm -- the dataframe of the yield advantages
    """
    Performance_abm = df_performance.copy()
    
    # Create a yield advantage feature 
    Performance_abm['yield_adv'] = (Performance_abm['c_yield'] - Performance_abm['o_yield'])/Performance_abm['c_yield']

    Performance_abm = Performance_abm[
            np.isinf(Performance_abm['yield_adv']) == False].reset_index(drop=True)
    
    return Performance_abm


def usda_acre_data(df, crop):
    """Merges the USDA county level yield data to the sales data.
    
    Keyword arguments:
        df -- the dataframe with sales data
        crop -- the crop we're adding acreage for
    Returns:
        df_w_acreage -- the dataframe with the yield data added
    """
    if crop == 'corn':
        ACRE_COUNTY_DATA = 'corn_acres.csv'
    elif crop == 'soybean':
        ACRE_COUNTY_DATA = 'soybean_acres.csv'
    
    # read in the yield data
    county_acres = pd.read_csv(DATA_DIR + ACRE_COUNTY_DATA, low_memory=False)
    
    # make the Value column numeric
    county_acres['Value'] = county_acres['Value'].str.replace(',', '').astype(float)
    
    # grab the year, values, state and county ANSIs
    county_acres_subset = county_acres[['Year', 'State ANSI', 'County ANSI', 'Value']]

    county_acres_avg = county_acres_subset.groupby(
            by=['Year', 'State ANSI'], as_index=False).mean()
    county_acres_avg = county_acres_avg[['Year', 'State ANSI', 'Value']]
    county_acres_avg = county_acres_avg.rename(columns={
            'Value': 'avg_' + crop +'_acres'})
    
    county_acres_subset = county_acres_subset.merge(county_acres_avg,
                                                    on=['Year', 'State ANSI'],
                                                    how='left')
    
    # change the county and state ANSIs to strings
    county_acres_subset['State ANSI'] = county_acres_subset['State ANSI'].astype(str)
    county_acres_subset['County ANSI'] = county_acres_subset['County ANSI'].astype(str)
    
    # strip the trailing .0s
    county_acres_subset[
            'County ANSI'] = county_acres_subset['County ANSI'].str.replace(r'.0$', '')
    
    county_acres_subset['County ANSI'] = np.where(
            county_acres_subset['County ANSI'].str.len() == 1,
            '00' + county_acres_subset['County ANSI'],
            county_acres_subset['County ANSI'])
    
    county_acres_subset['County ANSI'] = np.where(
            county_acres_subset['County ANSI'].str.len() == 2,
            '0' + county_acres_subset['County ANSI'],
            county_acres_subset['County ANSI'])
    
    county_acres_subset['fips'] = (county_acres_subset['State ANSI'] +
                       county_acres_subset['County ANSI'])
    
    # drop rows where county_yield_subset has nan
    nans = county_acres_subset[
            county_acres_subset['fips'].str.contains('nan')].index
    
    county_acres_subset = county_acres_subset.drop(nans).reset_index(drop=True)
    
    # make the county_yield_subset shipping fips numeric and merge with the sales df
    county_acres_subset['fips'] = pd.to_numeric(county_acres_subset['fips'])
    
    county_acres_subset = county_acres_subset[
            ['Year', 'Value', 'avg_' + crop +'_acres', 'fips']]
    
    county_acres_subset = county_acres_subset.rename(columns={'Year': 'year',
                                                              'Value': crop + '_acres'})
        
    county_acres_subset = county_acres_subset.drop_duplicates().reset_index(drop=True)
    
    # read in the abm map
    #abm_map = pd.read_csv(DATA_DIR + ABM_FIPS_MAP)
    abm_map = pd.read_csv(YEARLY_ABM_FIPS_MAP)
    
    abm_map = abm_map[['year', 'fips', 'abm']]
    
    # merge on abm
    county_acres_abm = county_acres_subset.merge(abm_map,
                                                 on=['year', 'fips'],
                                                 how='left')
    county_acres_abm = county_acres_abm.drop(columns=['fips'])
    county_acres_abm = county_acres_abm.groupby(by=['year', 'abm'], as_index=False).sum()
    county_acres_abm['year'] = county_acres_abm['year'].astype(str)
    
    # add the 2021 acres, setting them to 2020 for now
    county_acres_2021 = county_acres_abm[county_acres_abm['year'] == '2020'].reset_index(drop=True)
    county_acres_2021['year'] = '2021'
    county_acres_abm = pd.concat([county_acres_abm, county_acres_2021])
    
    # do the same for 2022
    county_acres_2022 = county_acres_2021.copy()
    county_acres_2022['year'] = '2022'
    county_acres_abm = pd.concat([county_acres_abm, county_acres_2022])
    
    df_w_acres = df.merge(county_acres_abm,
                          on=['year', 'abm'],
                          how='left')
    
    # same for 23! this is bad, have to think of a better way to make this work
    county_acres_2023 = county_acres_2022.copy()
    county_acres_2023['year'] = '2023'
    county_acres_abm = pd.concat([county_acres_abm, county_acres_2023])
    
    df_w_acres = df.merge(county_acres_abm,
                          on=['year', 'abm'],
                          how='left')
    
    return df_w_acres


def update_age_trait():
    """Updates the age/trait map
    
    Keyword arguments:
        None
    Returns:
        updated_map -- the updated age/trait map
    """
    CF_2022 = pd.read_excel(DATA_DIR + CF_2022_FILE)
    pred_set_index = CF_2022[
            CF_2022['FORECAST_YEAR'] == 2022].reset_index(drop=True)
    pred_set_index = pred_set_index[
            ['ACRONYM_NAME', 'BASE_TRAIT', 'TEAM_Y1_FCST_1']]
    pred_set_index = pred_set_index[
            pred_set_index['TEAM_Y1_FCST_1'] != 0].drop(columns=['TEAM_Y1_FCST_1']).reset_index(drop=True)
    
    # rename columns
    pred_set_index = pred_set_index.rename(columns={'BASE_TRAIT': 'trait', 
            'ACRONYM_NAME': 'Variety_Name'})
        
    # drop duplicates
    pred_set_index = pred_set_index.drop_duplicates().reset_index(drop=True)
    
    # read in the old age/trait map
    old_map = pd.read_csv('Age_Trait.csv')
    
    # get the first appearances of each hybrid from the index
    first_year_index = old_map['Variety_Name'].drop_duplicates().index
    first_year = old_map.iloc[first_year_index.values.tolist()].copy()
    
    # increment
    first_year['diff'] = 2023 - first_year['year']
    first_year['age'] = first_year['age'] + first_year['diff']
    
    # merge with the index
    pred_set_index_w_age = pred_set_index.merge(
            first_year.drop(columns=['year', 'trait', 'diff']),
            on=['Variety_Name'],
            how='left')
    
    # fill nas with 1
    pred_set_index_w_age['age'] = pred_set_index_w_age['age'].fillna(1).astype(int)
    pred_set_index_w_age['year'] = 2023
    
    updated_map = pd.concat([old_map, pred_set_index_w_age]).reset_index(drop=True)
    
    # read out
    updated_map.to_csv('Age_Trait_updated_2023.csv', index=False)
    
    return updated_map


def usda_yield_data(df):
    """Merges the USDA county level yield data to the sales data.
    
    Keyword arguments:
        df -- the dataframe with sales data
    Returns:
        df_w_yield -- the dataframe with the yield data added
    """
    # read in the yield data
    county_yield = pd.read_csv(DATA_DIR + YIELD_COUNTY_DATA, low_memory=False)
    
    # grab the year, values, state and county ANSIs
    county_yield_subset = county_yield[['Year', 'State ANSI', 'County ANSI', 'Value']]
    
    county_yield_avg = county_yield_subset.groupby(
            by=['Year', 'State ANSI'], as_index=False).mean()
    county_yield_avg = county_yield_avg[['Year', 'State ANSI', 'Value']]
    county_yield_avg = county_yield_avg.rename(columns={'Value': 'avg_yield'})
    
    county_yield_subset = county_yield_subset.merge(county_yield_avg,
                                                    on=['Year', 'State ANSI'],
                                                    how='left')
    
    # change the county and state ANSIs to strings
    county_yield_subset['State ANSI'] = county_yield_subset['State ANSI'].astype(str)
    county_yield_subset['County ANSI'] = county_yield_subset['County ANSI'].astype(str)
    
    # strip the trailing .0s
    county_yield_subset[
            'County ANSI'] = county_yield_subset['County ANSI'].str.replace(r'.0$', '')
    
    county_yield_subset['County ANSI'] = np.where(
            county_yield_subset['County ANSI'].str.len() == 1,
            '00' + county_yield_subset['County ANSI'],
            county_yield_subset['County ANSI'])
    
    county_yield_subset['County ANSI'] = np.where(
            county_yield_subset['County ANSI'].str.len() == 2,
            '0' + county_yield_subset['County ANSI'],
            county_yield_subset['County ANSI'])
    
    county_yield_subset['fips'] = (county_yield_subset['State ANSI'] +
                       county_yield_subset['County ANSI'])
    
    # drop rows where county_yield_subset has nan
    nans = county_yield_subset[
            county_yield_subset['fips'].str.contains('nan')].index
    
    county_yield_subset = county_yield_subset.drop(nans).reset_index(drop=True)
    
    # make the county_yield_subset shipping fips numeric and merge with the sales df
    county_yield_subset['fips'] = pd.to_numeric(county_yield_subset['fips'])
    
    county_yield_subset = county_yield_subset[
            ['Year', 'Value', 'avg_yield', 'fips']]
    
    county_yield_subset = county_yield_subset.rename(columns={'Year': 'year',
                                                              'Value': 'county_yield'})
    
    county_yield_subset = county_yield_subset.drop_duplicates().reset_index(drop=True)
    
    # read in the abm map
    abm_map = pd.read_csv(DATA_DIR + ABM_FIPS_MAP)
    abm_map = abm_map[['fips', 'abm']]
    
    # merge on abm
    county_yield_abm = county_yield_subset.merge(abm_map,
                                                 on=['fips'],
                                                 how='left')
    county_yield_abm = county_yield_abm.drop(columns=['fips'])
    
    county_yield_abm = county_yield_abm.groupby(by=['year', 'abm'], as_index=False).mean()
    county_yield_abm['year'] = county_yield_abm['year'].astype(str)
        
    county_yield_2021 = county_yield_abm[county_yield_abm['year'] == '2020'].reset_index(drop=True)
    county_yield_2021['year'] = '2021'
    county_yield_abm = pd.concat([county_yield_abm, county_yield_2021])

    # do the same for 2022
    county_yield_2022 = county_yield_2021.copy()
    county_yield_2022['year'] = '2022'
    county_yield_abm = pd.concat([county_yield_abm, county_yield_2022])

    df_w_yield = df.merge(county_yield_abm, 
                          on=['year', 'abm'],
                          how='left')
    
    return df_w_yield


def yield_aggregation(df):
    """Aggregates the yield by year/abm/hybrid.
    
    Keyword arguments:
        df -- the dataframe with the yield data
    Returns:
        df_with_yield
    """
    
    # selected required columns
    adv_abbr = df[['year', 'abm', 'c_hybrid', 'c_trait', 'c_yield']]
    
    df_with_yield = adv_abbr.groupby(by=['year', 'abm', 'c_hybrid', 'c_trait'],
                                     as_index=False).mean()
    df_with_yield = df_with_yield.rename(columns={'c_hybrid': 'hybrid',
                                                  'c_trait': 'trait',
                                                  'c_yield': 'yield'})
    return df_with_yield

