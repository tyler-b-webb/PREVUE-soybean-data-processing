#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:33:58 2022

@author: epnzv
"""

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
    for year in range(2008, 2021):
        print("Read ", str(year), " Sales Data")
        dfi_path = DATA_DIR + SALES_DIR + str(year) + '.csv'
        dfi = pd.read_csv(dfi_path)
        
        dfi = dfi[dfi['SPECIE_DESCR'] == 'SOYBEAN'].reset_index(drop=True)
        
        # set a year parameter to be the year 
        dfi['year'] = year
        
        # convert the effective date to a datetime format in order to set the mask 
        dfi['EFFECTIVE_DATE'] = pd.to_datetime(dfi['EFFECTIVE_DATE'])
        
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
    
    # group and sum
    Sale_2012_2020 = Sale_2012_2020.groupby(by=['year', 'Variety_Name', 'abm'],
                                            as_index=False).sum()
    Sale_2012_2020_full = Sale_2012_2020_full.groupby(by=['year', 'Variety_Name', 'abm'],
                                                      as_index=False).sum()
    
    # merge the eoy values
    Sale_2012_2020  = Sale_2012_2020.merge(Sale_2012_2020_full,
                                           on=['year', 'Variety_Name', 'abm'],
                                           how='left')
    
    # add the 2021 data
    Sale_2012_2021 = merge_2021_sales_data_impute_monthly(df=Sale_2012_2020,
                                                          abm_Teamkey=abm_Teamkey)
    
    # add the 2022 data
    Sale_2012_2022 = merge_2022_SCM_data(df=Sale_2012_2021,
                                         abm_Teamkey=abm_Teamkey)
    
    Sale_2012_2022 = Sale_2012_2022.fillna(0)
    
    # add in the 2023 prediction set
    Sale_2012_2023 = create_prediction_set(df=Sale_2012_2022,
                                           abm_Teamkey=abm_Teamkey)
    
    return Sale_2012_2023, Sale_2012_2020_full


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
