#!/usr/bin/env python
# coding: utf-8

# In[ ]:


LABEL_COLUMN = 'total_cases'
NUMERIC_COLUMNS = ['year',
                   'weekofyear',
                   'ndvi_ne',
                   'ndvi_nw',
                   'ndvi_se',
                   'ndvi_sw',
                   'precipitation_amt_mm',
                   'reanalysis_air_temp_k',
                   'reanalysis_avg_temp_k',
                   'reanalysis_dew_point_temp_k',
                   'reanalysis_max_air_temp_k',
                   'reanalysis_min_air_temp_k',
                   'reanalysis_precip_amt_kg_per_m2',
                   'reanalysis_relative_humidity_percent',
                   'reanalysis_sat_precip_amt_mm',
                   'reanalysis_specific_humidity_g_per_kg',
                   'reanalysis_tdtr_k',
                   'station_avg_temp_c',
                   'station_diur_temp_rng_c',
                   'station_max_temp_c',
                   'station_min_temp_c',
                   'station_precip_mm']
CATEGORICAL_COLUMNS = ['city']
CSV_COLUMNS = [LABEL_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
CSV_COLUMNS_NO_LABEL = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
CATEGORIES = {
    'city': ['sj', 'iq']
}

cols_to_norm = ['precipitation_amt_mm',
                'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k',
                'reanalysis_dew_point_temp_k',
                'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm',
                'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_tdtr_k',
                'station_avg_temp_c',
                'station_diur_temp_rng_c',
                'station_max_temp_c',
                'station_min_temp_c',
                'station_precip_mm']
cols_to_scale = ['year',
                 'weekofyear']

TRAIN_DATASET_FRAC = 0.8
DATETIME_COLUMN = "week_start_date"
train_file = 'dengue_features_train_1.csv'
test_file_in = 'dengue_features_test_in.csv'
test_file = 'dengue_features_test.csv'


# In[ ]:





# In[ ]:




