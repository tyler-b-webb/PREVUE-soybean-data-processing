#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:58:06 2021

@author: gmtxy
"""

import pandas as pd 
import numpy as np

from import_files import (read_2022_CF_data, read_2023_CF_data, read_2024_CF_data,
                          read_abm_teamkey_file,
                          read_commodity_corn_soybean, read_CY_CF_data, 
                          read_kynetic_data, read_performance,
                          read_sales_filepath, read_soybean_trait_map, read_SRP,
                          read_state_county_fips, read_weather_filepath, supply_data)
from merge import (merge_advantages, merge_cf_with_abm, merge_price_received)
from preprocess import (amend_trait_features, clean_commodity, clean_performance,
                        clean_state_county, clean_Weather, create_commodity_features,
                        create_imputation_frames, create_lagged_features, create_portfolio_weights,
                        create_lagged_sales, flatten_monthly_weather, get_RM,
                        impute_CY_CF, impute_h2h_data, impute_price, impute_SRP,
                        Performance_with_yield_adv, usda_acre_data, usda_yield_data)

###### --------------------- Read ABM & Teamkey Map  ------------------- ######
abm_Teamkey = read_abm_teamkey_file()

###### ---------------------- Read Sales Data ------------------------ ######
Sale_2012_2024 = read_sales_filepath(abm_Teamkey=abm_Teamkey)
Sale_2012_2024_lagged = create_lagged_sales(Sale_2012_2024)

Sale_all = get_RM(df=Sale_2012_2024_lagged)

print("Sale's Structure: ", Sale_all.info())
df_save_path = 'output/Sale_all.csv'
Sale_all.to_csv(df_save_path, index = False)
print("Check the fraction of missing values in Sales data: ", Sale_all.isna().sum())
print("Sale's shape: ", Sale_all.shape)

###### --------------------- Read Age & Trait Data -------------------- ######
Age_Trait = pd.read_csv('Age_Trait_2024.csv')

# set E3 to be XF WILL CHANGE LATER
Age_Trait['trait'] = Age_Trait['trait'].where(Age_Trait['trait'] != 'E3', 'XF')

Age_Trait['year'] = Age_Trait['year'].astype(dtype='str',copy=False)
print("Check the fraction of missing values in Age & Trait data: ", Age_Trait.isna().sum())
print("Age Trait's shape: ", Age_Trait.shape)

###### -------- Read Weather & County Location & FIPS_abm Data --------- ######
Weather_2012_2020, County_Location, FIPS_abm = read_weather_filepath()
print("Weather's Structure: ", Weather_2012_2020.info())
print("County_Location's Structure: ", County_Location.info())
print("FIPS_abm's Structure: ", FIPS_abm.info())

Weather = clean_Weather(Weather_2012_2020, County_Location, FIPS_abm) 
print("Weather's Structure: ", Weather.info())

Weather_Flattened = flatten_monthly_weather(Weather)

df_save_path = 'Flattened_Weather_abm_fips.csv'
Weather_Flattened.to_csv(df_save_path, index = False)
print("Flattened Weather's Structure: ", Weather_Flattened.info())
print("Check the fraction of missing value in weather data: ", Weather_Flattened.isnull().sum())
print("Flattened Weather's shape: ", Weather_Flattened.shape)

###### ------------------- Read Commodity Price Data ------------------ ######
Commodity_Corn, Commodity_Soybean = read_commodity_corn_soybean()

Commodity_Corn_Soybean = clean_commodity(Commodity_Corn, Commodity_Soybean)

CM_Soybean = create_commodity_features(Commodity_Corn_Soybean, 'soybean')
CM_Corn = create_commodity_features(Commodity_Corn_Soybean, 'corn')

# concatenate crops together
CM_Soybean_Corn = CM_Soybean.merge(CM_Corn, on=['year'])

CM_lagged = create_lagged_features(CM_Soybean_Corn)
df_save_path = 'CM_Soybean_Corn_Lagged.csv'
CM_lagged.to_csv(df_save_path, index = False)
print("Flattened Commodity_Price's Structure: ", CM_lagged.shape)

###### --------------------- Read Performance Data --------------------- ######
## State_County fips Files 
Performance_2011_2019 = read_performance()

State_fips, County_fips = read_state_county_fips()

State_County_abm = clean_state_county(State_fips, County_fips, FIPS_abm)
df_save_path = 'State_County_abm.csv'
State_County_abm.to_csv(df_save_path, index = False)

# Get abm level using fips
## H2H Files
###### ---------------- Read Consensus Forecasting Data ---------------- ######
CF_2016_2021 = pd.read_csv('CF_2016_2022.csv')

# drop 2021 data (it's in error, and add +1 to all years)
CF_2016_2021 = CF_2016_2021[CF_2016_2021['year'] != 2021].reset_index(drop=True)
CF_2016_2021['year'] = CF_2016_2021['year'] + 1

CF_2022 = read_2022_CF_data()
CF_2023 = read_2023_CF_data()
CF_2024 = read_2024_CF_data()
CF_2016_2022 = pd.concat([CF_2016_2021, CF_2022])
CF_2016_2023 = pd.concat([CF_2016_2022, CF_2023])
CF_2016_2024 = pd.concat([CF_2016_2023, CF_2024])

#CF_2016_2022 = read_concensus_forecasting()
# get the current year data and merge it


CF_2016_2024['year'] = CF_2016_2024['year'].astype(int).astype(str)

# drop missing value 
CF_2016_2024 = CF_2016_2024.dropna(how='any')

#print("Concensus Forecasting data's Structure: ", CF_2016_2022.info())
#print("Checking the fraction of missing value: ", CF_2016_2022.isna().sum()/CF_2016_2022.shape[0])
df_save_path = 'CF_2016_2023.csv'

#CF_2016_2022.to_csv(df_save_path, index = False)
CF_abm = merge_cf_with_abm(CF_2016_2024, abm_Teamkey)
CF_abm['year'] = CF_abm['year'].astype(dtype='str', copy=False)

# aggregate to get rid of 0 weirdness
CF_abm = CF_abm.groupby(
        by=['year', 'Variety_Name', 'abm'], as_index=False).sum().reset_index(drop=True)

###### --------------------------- Read trait map  ------------------------- ######
trait_map = read_soybean_trait_map()
SRP_2011_2024 = read_SRP()

##### ---------------------------- Read Kynetic Data ---------------------######
kynetic_data = read_kynetic_data()

###### ----------------------- Merge All Datasets ---------------------- ######
def merge_all():
    """ Reads in and returns the final combined dataframe.
    
    Keyword arguments:
        None
    Returns:
        Sale_HP_trait_weather_CM_Performance_CF -- the dataframe of sales, 
                                                    hot products, trait/age, 
                                                    weather, commoidty price, 
                                                    concensus forecasting data
    """
    # ## Merge Sale with HP
    # print("Step 1: Merge Sale_2012_2019 with Hot Products......")
    # Sale_HP = Sale_2012_2019_lagged.merge(Hot_Products, how = 'left', on = ['year', 'Variety_Name'])
    # #print("Step 1: Sale_HP's structure: ", Sale_HP.info())
    # print("Step 1: Sale_HP's shape: ", Sale_HP.shape)
    # print("..................")
    
    ## Merge Sale_HP with age_trait 
    print("Step 2: Merge Sale_HP with Age_Trait......")
    Sale_HP_trait = Sale_all.merge(Age_Trait, how = 'left', on = ['year', 'Variety_Name'])
    if 'Unnamed: 18' in Sale_HP_trait.columns:
        print('point 1')
        
    #print("Step 2: Sale_HP_trait's structure: ", Sale_HP_trait.info())
    print("Step 2: Sale_HP_trait's shape: ", Sale_HP_trait.shape)
    print("..................")
    
    # impute lagged sales for age one products
    #
    #Sale_HP_trait = impute_age_one_lagged(df=Sale_HP_trait)
    
    
    ## Merge Sale_HP_trait with weather_flattened
    print("Step 3: Merge Sale_HP_trait with Weather_flattened......")
    Sale_HP_trait_weather = Sale_HP_trait.merge(Weather_Flattened, how = 'left', on = ['year', 'abm'])
    if 'Unnamed: 18' in Sale_HP_trait_weather.columns:
        print('point 2')
    #print("Step 3: Sale_HP_trait_weather's structure: ", Sale_HP_trait_weather.info())
    print("Step 3: Sale_HP_trait_weather's shape: ", Sale_HP_trait_weather.shape)
    print("..................")
    
    ## Merge Sale_HP_trait_weather with CM_Soybean_Corn
    print("Step 4: Merge Sale_HP_trait with CM_lagged_soybean_corn......")
    Sale_HP_trait_weather_CM = Sale_HP_trait_weather.merge(CM_lagged, how = 'left', on = ['year'])
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM.columns:
        print('point 3')
    print("Step 4: Sale_HP_trait_weather_CM's shape: ", Sale_HP_trait_weather_CM.shape)
    print("..................")
    
    
    ## Merge Sale_HP_trait_weather_CM with the Performance
    print("Step 5: Merge Sale_HP_trait_weather_CM with Performance......")

    Performance_abm = Performance_2011_2019.merge(State_County_abm, how = 'left', on = ['state', 'county'])   
    Performance_yield_adv = Performance_with_yield_adv(Performance_abm)    
    Performance_adv = merge_advantages(Performance_yield_adv)
    
    Performance_adv = clean_performance(Performance_adv)
    df_save_path = 'Performance_adv.csv'
    Performance_adv.to_csv(df_save_path, index = False)
    
    product_abm_level, trait_abm_year_level, abm_year_level, year_level = create_imputation_frames(
                df=Performance_adv)
    # rename hybrid columns
    Performance_adv1 = Performance_adv.rename(columns = {'hybrid': 'Variety_Name'})
    # drop trait columns
    Performance_adv1 = Performance_adv1.drop(columns=['trait'])
    Performance_adv1['year'] = Performance_adv1['year'].astype(str)
    
    Sale_HP_trait_weather_CM_Performance = Sale_HP_trait_weather_CM.merge(Performance_adv1,
                                        on=['year', 'abm', 'Variety_Name'],
                                        how='left')
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance.columns:
        print('point 4')
    # impute the missing value 
    Sale_HP_trait_weather_CM_Performance = impute_h2h_data(Sale_HP_trait_weather_CM_Performance, 
                                                            product_abm_level, trait_abm_year_level,
                                                            abm_year_level, year_level)
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance.columns:
        print('point 5')
    
    # replace any blank trait values with "Conventional"
    Sale_HP_trait_weather_CM_Performance['trait'] = Sale_HP_trait_weather_CM_Performance['trait'].fillna('Conventional')
    print("Step 5: Sale_HP_trait_weather_CM's shape: ", Sale_HP_trait_weather_CM_Performance.shape)
    print("..................")
    
    df_save_path = 'Sale_HP_trait_weather_CM_Performance.csv'
    Sale_HP_trait_weather_CM_Performance.to_csv(df_save_path, index = False)
    
    ## Merge Sale_HP_trait_weather_CM_Performance with Concensus Forecasting
    print("Step 6: Merge Sale_HP_trait_weather_CM with CF and kynetic data......")
    Sale_HP_trait_weather_CM_Performance_CF = Sale_HP_trait_weather_CM_Performance.merge(
            CF_abm, how = 'left', on = ['year','Variety_Name','abm'])
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance_CF.columns:
        print('point 6')
    Sale_HP_trait_weather_CM_Performance_CF = Sale_HP_trait_weather_CM_Performance_CF.merge(
            kynetic_data, how='left', on=['year', 'Variety_Name', 'abm'])
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance_CF.columns:
        print('point 7')
    print("Step 6: Sale_HP_trait_weather_CM's shape: ", Sale_HP_trait_weather_CM_Performance_CF.shape)
    print("..................")
    
    print("Checking the portion of missing value in the combined dataset: ")
    print(Sale_HP_trait_weather_CM_Performance_CF.isna().sum()/Sale_HP_trait_weather_CM_Performance_CF.shape[0])
    Sale_HP_trait_weather_CM_Performance_CF['TEAM_Y1_FCST_1'] = Sale_HP_trait_weather_CM_Performance_CF['TEAM_Y1_FCST_1'].fillna(0)
    
    print("Step 7: Merge Sale_HP_trait_weather_CM_CF with SRP......")
    Sale_HP_trait_weather_CM_Performance_CF_SRP = Sale_HP_trait_weather_CM_Performance_CF.merge(SRP_2011_2023, how = 'left', on = ['year', 'Variety_Name'])
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance_CF_SRP.columns:
        print('point 8')
    # impute the missing value
    Sale_HP_trait_weather_CM_Performance_CF_SRP = impute_SRP(Sale_HP_trait_weather_CM_Performance_CF_SRP)
    
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance_CF_SRP.columns:
        print('point 9')
    
    # impute missing price values as well
    Sale_HP_trait_weather_CM_Performance_CF_SRP = impute_price(Sale_HP_trait_weather_CM_Performance_CF_SRP)
    
    if 'Unnamed: 18' in Sale_HP_trait_weather_CM_Performance_CF_SRP.columns:
        print('point 10')
    
    print("Step 7: Sale_HP_trait_weather_CM_SRP's shape: ", Sale_HP_trait_weather_CM_Performance_CF_SRP.shape)
    print("..................")
    
    print("Saving Files.........")
    df_save_path = 'Sale_HP_trait_weather_CM_Performance_CF_SRP.csv'
    Sale_HP_trait_weather_CM_Performance_CF_SRP.to_csv(df_save_path, index = False)

    return Sale_HP_trait_weather_CM_Performance_CF_SRP, Sale_HP_trait_weather

Sale_HP_trait_weather_CM_Performance_CF_SRP, Sale_HP_trait_weather = merge_all()

# encoding trait
print("Step 8: Encoding trait")
Final_df = Sale_HP_trait_weather_CM_Performance_CF_SRP.merge(trait_map,
                                                             how='left',
                                                             on=['trait']) 


# add county yield data
sales_w_county_yield = usda_yield_data(df=Final_df)

# add the USDA acreage data
sales_w_corn_acreage = usda_acre_data(df=sales_w_county_yield, crop='corn')
sales_w_soybean_acreage = usda_acre_data(df=sales_w_corn_acreage, crop='soybean')

sales_w_soybean_acreage = sales_w_soybean_acreage.replace(-np.inf, 0)
sales_w_soybean_acreage = sales_w_soybean_acreage.replace(np.inf, 0)
sales_w_soybean_acreage = sales_w_soybean_acreage.fillna(0)
sales_w_soybean_acreage.loc[sales_w_soybean_acreage['age'] == 0, 'age'] = 1

Final_df_acreage = sales_w_soybean_acreage.copy()#drop(columns = ['trait'])

# drop any UNKNOWNs
Final_df_acreage = Final_df_acreage.rename(columns={'Variety_Name': 'hybrid'})
Final_df_acreage = Final_df_acreage[
        Final_df_acreage['hybrid'] != 'UNKNOWN'].reset_index(drop=True)
Final_df_acreage = Final_df_acreage[
        Final_df_acreage['abm'] != 'UNK'].reset_index(drop=True)

# set the 'pred_price' feature to be the price feature
Final_df_acreage['pred_price'] = Final_df_acreage['price'].copy()

# get the avai_supply_region
Final_df_acreage = supply_data(df=Final_df_acreage)

# get the price_rec data
Final_df_acreage = merge_price_received(df=Final_df_acreage)
    
# edit trait columns
Final_df_acreage = amend_trait_features(df=Final_df_acreage)

# create the product weights
weights_matrix = create_portfolio_weights(df=Final_df_acreage)

# drop any columns we aren't interested in
Final_df_acreage = Final_df_acreage.drop(
        columns=['discount', 'Unnamed: 18', 'county_yield', 'avg_yield', 'corn_acres',
                 'avg_corn_acres', 'soybean_acres', 'avg_soybean_acres'])


df_save_path = 'training_data_set_2024_feb28.csv'

Final_df_acreage = Final_df_acreage.drop_duplicates()

# set the 2023 nets_Q_eoy to be the forecast from soy success
net_sales_23_fcst = pd.read_csv('net_sales_23_fcst.csv')
net_sales_23_fcst['year'] = net_sales_23_fcst['year'].astype(str)

Final_df_acreage = Final_df_acreage.merge(net_sales_23_fcst,
                                          on=['year', 'hybrid', 'abm'],
                                          how='left')

Final_df_acreage.loc[
        Final_df_acreage['year'] == '2023', 'nets_Q_eoy'] =  Final_df_acreage.loc[
                Final_df_acreage['year'] == '2023', 'loc'].values

Final_df_acreage['nets_Q'] = Final_df_acreage['nets_Q_eoy'].values
Final_df_acreage = Final_df_acreage.drop(columns=['nets_Q_eoy', 'loc'])

Final_df_acreage = Final_df_acreage.fillna(0)

Final_df_acreage.to_csv(df_save_path, index = False)
Final_df_acreage.isna().sum()

Final_df_acreage.isna().sum().sum()

Final_df_acreage