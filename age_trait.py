#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:59:07 2022

@author: epnzv
"""

import numpy as np
import pandas as pd
import re

from aggregation_config import (CF_2022_FILE, DATA_DIR, SALES_2021, SCM_DATA_DIR,
                                SCM_DATA_FILE)

old_at = pd.read_csv('Age_Trait_2023_fixed.csv')
    
# read in the traits and hybrids for 21 and 22
sales_21 = pd.read_csv(DATA_DIR + SALES_2021)
sales_21 = sales_21[['VARIETY', 'Trait']].drop_duplicates().reset_index(drop=True)

sales_22 = pd.read_csv(DATA_DIR + SCM_DATA_DIR + SCM_DATA_FILE)
sales_22 = sales_22[['VARIETY', 'Trait']].drop_duplicates().reset_index(drop=True)

CF_2022 = pd.read_excel(DATA_DIR + CF_2022_FILE)
sales_23 = CF_2022[CF_2022['FORECAST_YEAR'] == 2022].reset_index(drop=True)
sales_23 = sales_23[['ACRONYM_NAME', 'TEAM_Y1_FCST_1', 'TRAIT_NAME']]
sales_23 = sales_23[
        sales_23['TEAM_Y1_FCST_1'] != 0].drop(columns=['TEAM_Y1_FCST_1']).reset_index(drop=True)


sales_23 = sales_23.rename(columns={'ACRONYM_NAME': 'Variety_Name',
                                    'TRAIT_NAME': 'trait'})
sales_23['year'] = 2023
sales_23.loc[sales_23['trait'] == 'CONV', 'trait'] = 'Conventional'

CF_2023 = pd.read_csv(DATA_DIR + 'product_list_zones_24.csv')
sales_24 = CF_2023[['ACRONYM_NAME', 'BASE_TRAIT']]
sales_24 = sales_24.rename(columns={'ACRONYM_NAME': 'Variety_Name',
                                    'BASE_TRAIT': 'trait'})
sales_24['year'] = 2024
sales_24.loc[sales_24['trait'] == 'CONV', 'trait'] = 'Conventional'


traits_to_drop = ['RR2X/DC/XF', 'RR2X/DC/SR', 'RR2X/DC/SR/XF']

# drop the empty traits
for trait in traits_to_drop:
    sales_21 = sales_21[sales_21['Trait'] != trait].reset_index(drop=True)
    sales_22 = sales_22[sales_22['Trait'] != trait].reset_index(drop=True)
    
# rename stuff
rename_dic = {'HT3': 'XF', 'RR2 XTEND': 'RR2X', 'CONV': 'Conventional', 'HT3/SR':
    'XF/SR'}
    
for trait in rename_dic.keys():
    sales_21['Trait'] = np.where(sales_21['Trait'] == trait,
            rename_dic[trait],
            sales_21['Trait'])
    sales_22['Trait'] = np.where(sales_22['Trait'] == trait,
            rename_dic[trait],
            sales_22['Trait'])
    
# fill Emptys based on name (string parse)
empty_hybrids_21 = {'AG55XF0': 'XF/SR', 'AG69XF0': 'XF/SR', 'AG72XF0': 'XF/SR'}

empty_hybrids_22 = {}
for variety in sales_22.loc[sales_22['Trait']=='(Empty)', 'VARIETY'].unique():
    if 'XF' in variety:
        temp = re.findall(r'\d+', variety)
        res = list(map(int, temp))
        sr_digit = res[0]
        if sr_digit < 45:
            empty_hybrids_22[variety] = 'XF'
        elif sr_digit >= 45:
            empty_hybrids_22[variety] = 'XF/SR'
            
for hybrid in empty_hybrids_21.keys():
    sales_21.loc[sales_21['VARIETY'] == hybrid, 'Trait'] = empty_hybrids_21[hybrid]

for hybrid in empty_hybrids_22.keys():
    sales_22.loc[sales_22['VARIETY'] == hybrid, 'Trait'] = empty_hybrids_22[hybrid]
    
# drop anything with Empty in it
sales_21 = sales_21[sales_21['Trait'] != '(Empty)'].reset_index(drop=True)
sales_22 = sales_22[sales_22['Trait'] != '(Empty)'].reset_index(drop=True)

# drop any duplicates
sales_21 = sales_21.drop_duplicates().reset_index(drop=True)
sales_22 = sales_22.drop_duplicates().reset_index(drop=True)

sales_21 = sales_21.rename(columns={'VARIETY': 'Variety_Name',
                                    'Trait': 'trait'})
sales_22 = sales_22.rename(columns={'VARIETY': 'Variety_Name',
                                    'Trait': 'trait'})

# now go ahead and calculate the ages for the 21 and 22
sales_21['year'] = 2021
sales_22['year'] = 2022

hybrid_first_year = pd.DataFrame(columns=['Variety_Name', 'first_year'])

for hybrid in old_at['Variety_Name'].unique():
    first_year = min(old_at.loc[old_at['Variety_Name'] == hybrid, 'year'])
        
    data={'Variety_Name': [hybrid],
          'first_year': [first_year]}
    
    single_hybrid = pd.DataFrame.from_dict(data)
        
    if hybrid_first_year.empty == True:
        hybrid_first_year = single_hybrid.copy()
    else:
        hybrid_first_year = pd.concat([hybrid_first_year, single_hybrid]).reset_index(drop=True)
    
sales_21 = sales_21.drop_duplicates().reset_index(drop=True)
sales_22 = sales_22.drop_duplicates().reset_index(drop=True)
sales_23 = sales_23.drop_duplicates().reset_index(drop=True)
sales_24 = sales_24.drop_duplicates().reset_index(drop=True)

sales_21 = sales_21.merge(hybrid_first_year, on=['Variety_Name'], how='left')
sales_22 = sales_22.merge(hybrid_first_year, on=['Variety_Name'], how='left')
sales_23 = sales_23.merge(hybrid_first_year, on=['Variety_Name'], how='left')
sales_24 = sales_24.merge(hybrid_first_year, on=['Variety_Name'], how='left')

sales_21['age'] = sales_21['year'] - sales_21['first_year'] + 1
sales_22['age'] = sales_22['year'] - sales_22['first_year'] + 1
sales_23['age'] = sales_23['year'] - sales_23['first_year'] + 1
sales_24['age'] = sales_24['year'] - sales_24['first_year'] + 1

# fill any nas with 1
sales_21['age'] = sales_21['age'].fillna(1)
sales_22['age'] = sales_22['age'].fillna(1)
sales_23['age'] = sales_23['age'].fillna(1)
sales_24['age'] = sales_24['age'].fillna(1)


# drop first year column and concat
sales_21 = sales_21[['year', 'Variety_Name', 'age', 'trait']]
sales_22 = sales_22[['year', 'Variety_Name', 'age', 'trait']]
sales_23 = sales_23[['year', 'Variety_Name', 'age', 'trait']]
sales_24 = sales_24[['year', 'Variety_Name', 'age', 'trait']]

old_at = old_at[old_at['year']!= 2021].reset_index(drop=True)
old_at = old_at[old_at['year']!= 2022].reset_index(drop=True)

new_at = pd.concat([old_at, sales_21]).reset_index(drop=True)
new_at = pd.concat([new_at, sales_22]).reset_index(drop=True)
new_at = pd.concat([new_at, sales_23]).reset_index(drop=True)
new_at = pd.concat([new_at, sales_24]).reset_index(drop=True)

new_at.to_csv('Age_Trait_2024.csv', index=False)