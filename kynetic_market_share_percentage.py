#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:04:11 2022

@author: epnzv
"""

import pandas as pd
import numpy as np

kynetic_data = pd.read_csv('../../NA-soy-pricing/data/soybean_kynetic_2008_2022.csv')

brand_groups = pd.read_csv('../../NA-soy-pricing/data/2022_seed_brand_groups.csv')

kynetic_2022 = kynetic_data[kynetic_data['Year'] == 2022].reset_index(drop=True)[[
        'Crop', 'Company/Brand', 'Seed Trait', 'Projected Acres']]

kynetic_2022_soybeans = kynetic_2022[kynetic_2022['Crop'] == 'Soybeans'].reset_index(drop=True)

kynetic_2022_soybeans['Projected Acres'] = kynetic_2022_soybeans[
        'Projected Acres'].str.replace(',','').astype(int)

total_acres = np.sum(kynetic_2022_soybeans['Projected Acres'])

kynetic_2022_group = kynetic_2022_soybeans.merge(
        brand_groups, on=['Company/Brand'], how='left').drop(columns=['Company/Brand'])


by_company = kynetic_2022_soybeans.groupby(by=['Company/Brand'], as_index=False).sum()

by_trait = kynetic_2022_soybeans.groupby(by=['Seed Trait'], as_index=False).sum()

by_group = kynetic_2022_group.groupby(by=['Group'], as_index=False).sum()

by_company['percentage'] = by_company['Projected Acres'] / total_acres
by_trait['percentage'] = by_trait['Projected Acres'] / total_acres
by_group['percentage'] = by_group['Projected Acres'] / total_acres

by_company = by_company.sort_values(by=['percentage'], ascending=False).reset_index(drop=True)
by_trait = by_trait.sort_values(by=['percentage'], ascending=False).reset_index(drop=True)
by_group = by_group.sort_values(by=['percentage'], ascending=False).reset_index(drop=True)


by_company.to_csv('market_share_22_by_company.csv', index=False)
by_trait.to_csv('market_share_22_by_trait.csv', index=False)
by_group.to_csv('market_share_22_by_group.csv', index=False)

