import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn_pandas import DataFrameMapper
warnings.filterwarnings('ignore', category=DataConversionWarning)

def transform_date(df, field_name, drop=True):
    field = df[field_name]
    if not np.issubdtype(field, np.datetime64):
        df[field_name] = field = pd.to_datetime(field, infer_datetime_format=True)
    target_pre = re.sub('[Dd]ate$', '', field_name)
    for i in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[target_pre + i] = getattr(field.dt, i.lower())
    df[target_pre + 'Elapsed'] = field.astype(np.int64) // 10**9
    if drop:
        df.drop(field_name, axis=1, inplace=True)


def create_category_fields(df, is_train=True, train_df=None):  
    if is_train:
        for col_name, data in df.items():
            if is_string_dtype(data):
                df[col_name] = data.astype('category').cat.as_ordered()
    else:
        for col_name, data in df.items():
            if (col_name in train_df.columns) and (train_df[col_name].dtype.name == 'category'):
                df[col_name] = pd.Categorical(data, categories=train_df[col_name].cat.categories, ordered=True)



def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique() > max_n_cat):
        df[name] = col.cat.codes + 1


def scale_vars(df, mapper):
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper


def preprocessing(df, y_fld, skip_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, mapper=None):

    if not skip_flds: 
        skip_flds = []
    df = df.copy()
    if preproc_fn: 
        preproc_fn(df)

    y = df[y_fld].values
    df.drop(skip_flds + [y_fld], axis=1, inplace=True)

    if na_dict is None: 
        na_dict = {}
    for n,c in df.items(): 
        na_dict = fix_missing(df, c, n, na_dict)
    if do_scale: 
        mapper = scale_vars(df, mapper)
    for n,c in df.items(): 
        numericalize(df, c, n, max_n_cat)

    res = [pd.get_dummies(df, dummy_na=True), y, na_dict]

    if do_scale: 
        res = res + [mapper]
    return res

def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))