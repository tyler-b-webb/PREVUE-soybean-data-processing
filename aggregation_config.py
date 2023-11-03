#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:19:38 2021

@author: epnzv
"""

ABM_FIPS_MAP = 'mappingall_processed.csv'

BIG_CF_FILE = 'Soybean_CY_Asgrow_12_29_21.csv'

BLIZZARD_DIR = '../../NA-soy-pricing/dataframe_construction_r_r/blizzard/county_data/'

CF_2022_FILE = 'FY23_01_20_22.xlsx'

CF_2023_FILE = 'FY23_Soy_011923.xlsx'

CM_DIR = 'CM_prep/'

CORN_SOY_ACRES = 'acres_corn_soy_08_to_19.csv'

# the data directory
DATA_DIR = '../../NA-soy-pricing/data/'

H2H_DIR = 'H2H_yield_data/'

HISTORICAL_SRP = 'historical_SRP/'

HISTORICAL_SUPPLY = 'hist_supply_info.csv'

KYNETIC_DATA = 'soybean_kynetic_2008_2022.csv'

# the price received data file
PRICE_REC = 'price_received_06to22Dec.csv'

PROD_LIST_23 = '2023_prelim_asgrow_prod_list.csv'

PROD_LIST_24 = 'product_list_zones_24.csv'

SCM_DATA_DIR = 'SCM_data/'

SCM_DATA_FILE = 'may11_22_SCM.csv'

# the abm table
ABM_TABLE = 'ABM_Table.csv'

E3_EQUAL_XF = True

EFFECTIVE_DATE = {'month': 2,
                  'day': 28}

DAILY_FRACTIONS = 'historical_daily_fractions_all_features.csv'

MONTHLY_FRACTIONS = 'historical_monthly_fractions.csv'

# the list of columns to drop from the kynetic data
KYNETIC_COLUMNS_TO_DROP = ['Crop', 'Respondent', 'State (Numeric)', 'State',
                           'Acre Range', 'Company/Brand', 'Seed Trait', 
                           'Projected Units', 'Total Pounds',
                           'Projected Free Units',
                           'Projected Dollars',
                           'Projected Acres', 'Total Seeds']

# the dictionary of the column names we want to use for the kynetic data (renaming)
KYNETIC_COLUMN_NAMES = {'County (Numeric)': 'fips', 'Hybrid/Variety': 'Variety_Name',
                        'Year': 'year', 'Retail Price': 'price',
                        'Discount Amount': 'discount'}

SALES_2021 = '2021_hybrid_abm_dealer.csv'

SALES_2022 = '2022_hybrid_abm_dealer.csv'

SALES_2021_W_DATE = 'D1_MS_21_product_location_202110281734.csv'

#SALES_2022 = 'nov1_21_order_bank.csv'

SALES_DIR = 'sales_data/'

OLD_2020 = '2020_old.csv'

YIELD_COUNTY_DATA = 'county_soybean_yield.csv'

ORDER_DATE = {'month': 12,
              'day': 15}

# the weight to use to get the orders to date for 2021, where we don't have effective dates
ORDER_FRACTION_2021 = 0.36

US_STATE_ABBREV = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
    }


YEARLY_ABM_FIPS_MAP = 'abm_years_08_to_22.csv'