#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:12:53 2023

@author: epnzv
"""

import pandas as pd 
import numpy as np

from aggregation_config import (DATA_DIR, SALES_DIR)
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
# define a list to store the FULL datasets
dfs_full = []

for year in range(2012, 2021):
    print("Read ", str(year), " Sales Data")
    dfi_path = DATA_DIR + SALES_DIR + str(year) + '.csv'
    dfi = pd.read_csv(dfi_path)
    
    dfi = dfi[dfi['SPECIE_DESCR'] == 'SOYBEAN'].reset_index(drop=True)
    dfi = dfi[dfi['BRAND_FAMILY_DESCR'] == 'NATIONAL'].reset_index(drop=True)
    
    # set a year parameter to be the year 
    dfi['year'] = year
    
    dfs_full.append(dfi.copy())
      
# concate all dataframes  

Sale_2012_2020_full = pd.concat(dfs_full).reset_index(drop=True)

Sale_2012_2020_full = Sale_2012_2020_full.rename(
        columns={'SHIPPING_FIPS_CODE': 'fips',
                 'SLS_LVL_2_ID': 'abm',
                 'NET_SALES_QTY_TO_DATE': 'net_sales'})
    
Sale_2012_2020_full = Sale_2012_2020_full[['year', 'abm', 'fips', 'net_sales']]

abm_map_sales = pd.DataFrame()

for year in Sale_2012_2020_full['year'].unique():
    single_year = Sale_2012_2020_full[
            Sale_2012_2020_full['year'] == year].reset_index(drop=True)
    
    for fips in single_year['fips'].unique():
        single_year_fips = single_year[
                single_year['fips'] == fips].reset_index(drop=True)
        
        single_year_fips_agg = single_year_fips.groupby(
                by=['year', 'abm', 'fips'], as_index=False).sum()
        
        if len(single_year_fips_agg) > 1:
            single_year_fips_agg = single_year_fips_agg.sort_values(
                    by=['net_sales'], ascending=False).reset_index(drop=True)
            single_year_fips_agg = single_year_fips_agg.head(1)
        
        if abm_map_sales.empty == True:
            abm_map_sales = single_year_fips_agg.copy()
        else:
            abm_map_sales = pd.concat(
                    [abm_map_sales, single_year_fips_agg]).reset_index(drop=True)
            
abm_map = abm_map_sales.drop(columns=['net_sales'])

# import old mapping
YEARLY_ABM_FIPS_MAP = 'abm_years_08_to_22.csv'
FIPS_abm_Address = YEARLY_ABM_FIPS_MAP #DATA_DIR + 'abm_years.csv'
FIPS_abm = pd.read_csv(FIPS_abm_Address)

FIPS_abm_area_id = FIPS_abm[
        ['New Area ID', 'abm']].copy().drop_duplicates().reset_index(drop=True)

abm_map_area_id = abm_map.merge(FIPS_abm_area_id, on=['abm'], how='left')

abm_map_area_id_concat = pd.concat(
        [abm_map_area_id, FIPS_abm[FIPS_abm['year'] > 2020]]).reset_index(drop=True)

abm_map_no_dupes = pd.DataFrame()

for year in abm_map_area_id_concat['year'].unique():
    single_year = abm_map_area_id_concat[
            abm_map_area_id_concat['year'] == year].reset_index(drop=True)
    
    for fips in single_year['fips'].unique():
        single_year_fips = single_year[
                single_year['fips'] == fips].reset_index(drop=True)
        
        if len(single_year_fips) > 1:
            single_year_fips = single_year_fips.head(1)
            
        if abm_map_no_dupes.empty == True:
            abm_map_no_dupes = single_year_fips.copy()
        else:
            abm_map_no_dupes = pd.concat(
                    [abm_map_no_dupes, single_year_fips]).reset_index(drop=True)
            
abm_map_no_dupes.loc[abm_map_no_dupes['abm'] == '9Z01', 'New Area ID'] = '9Z01'
abm_map_no_dupes.loc[abm_map_no_dupes['abm'] == 'UNK', 'New Area ID'] = 'UNK'

abm_map_no_dupes_dropped = abm_map_no_dupes.dropna()

abm_map_no_dupes_dropped.to_csv('soybean_abm_fips_map.csv', index=False)