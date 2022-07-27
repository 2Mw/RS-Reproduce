import datetime
import gc
from cf.utils.logger import logger

import pandas as pd
from dateutil.relativedelta import relativedelta

import cf.preprocess.data as base
import numpy as np
import os

"""
The baseline method of preprocess data:  https://github.com/timberding/CTR-prediction-through-cross-domain-data-from-ads-and-news-feeds
# Competition Link: https://developer.huawei.com/consumer/cn/activity/starAI2022/algo/competition.html
# dataset link: https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/2022/AI/238143c8f2b211ec84305c80b6c7dac5/2022_3_data.zip
"""

# =========================   Target domain  train_data_ads.csv =========================

ADS_SPARSE_FEATURE_NAME = ['user_id', 'gender', 'residence', 'city', 'city_rank', 'series_dev', 'series_group',
                           'emui_dev', 'device_name', 'net_type', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id',
                           'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id', 'hispace_app_tags',
                           'app_second_class']

ADS_DENSE_FEATURE_NAME = ['age', 'device_size', 'app_score']

ADS_SEQUENCE_FEATURE_NAME = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003', 'ad_close_list_v001',
                             'ad_close_list_v002', 'ad_close_list_v003']

# =========================   Source domain  train_data_feeds.csv =========================

FEEDS_SPARSE_FEATURE_NAME = ['u_browserLifeCycle', 'u_browserMode', 'u_feedLifeCycle']

FEEDS_DENSE_FEATURE_NAME = ['u_phonePrice', 'u_refreshTimes']

FEEDS_SEQUENCE_FEATURE_NAME = ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news']

# ==== Summary

DENSE_NAMES = FEEDS_DENSE_FEATURE_NAME + ADS_DENSE_FEATURE_NAME
SPARSE_NAMES = ADS_SPARSE_FEATURE_NAME + FEEDS_SPARSE_FEATURE_NAME
SEQ_NAMES = FEEDS_SEQUENCE_FEATURE_NAME + ADS_SEQUENCE_FEATURE_NAME
# Other

SEP = ','
SEQ_SPLIT = '^'


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    """
    Create Huawei dataset
    :param file:
    :param sample_num:
    :param test_size:
    :param numeric_process:
    :return:
    """
    seq_names, seq_map = ADS_SEQUENCE_FEATURE_NAME + FEEDS_SEQUENCE_FEATURE_NAME, {}
    dirname = os.path.dirname(file)
    files = ['train_data_ads.csv', 'train_data_feeds.csv', '../test/test_data_ads.csv']
    ads_file, feeds_file, test_file = [os.path.join(dirname, i) for i in files]
    # read
    train_data_ads_df = base.read_data(ads_file, sample_num, SEP, None, dtype=str)
    train_data_feeds_df = base.read_data(feeds_file, sample_num, SEP, None, dtype=str)
    test_data_df = base.read_data(os.path.realpath(test_file), sample_num, SEP, None, dtype=str)
    # Export test log_id info
    test_data_df['log_id'].to_csv(os.path.join(os.path.dirname(dirname), 'log_id.csv'), index=False)
    # ====== preprocess ads data
    ads_feature = ADS_SPARSE_FEATURE_NAME + ADS_DENSE_FEATURE_NAME + ADS_SEQUENCE_FEATURE_NAME
    # train_data_ads_df['date_flg'] = train_data_ads_df.pt_d.str[0:8]
    # test_data_df['date_flg'] = test_data_df.pt_d.str[0:8]
    train_data_ads_df = train_data_ads_df[ads_feature + ['label']]
    test_data_df = test_data_df[ads_feature]
    # ====== preprocess feeds data
    # process date
    # ! maybe can ignore date preprocess
    # train_data_feeds_df['join_date'] = train_data_feeds_df.e_et.str[0:8]
    # train_data_feeds_df['join_date'] = train_data_feeds_df['join_date'].apply(
    #     lambda x: (datetime.datetime.strptime(x, '%Y%m%d') + relativedelta(days=1)).strftime('%Y%m%d'))
    # remove overlapped and redundant feeds columns
    # ! maybe we can store other columns like `i_xxx`
    # ! maybe can ignore join_date column
    feeds_feature = FEEDS_SPARSE_FEATURE_NAME + FEEDS_DENSE_FEATURE_NAME + FEEDS_SEQUENCE_FEATURE_NAME + [
        'u_userId']
    train_data_feeds_df = train_data_feeds_df[feeds_feature]
    train_data_feeds_df = train_data_feeds_df.groupby(['u_userId'], as_index=False).max()

    # ====== aggregation data
    train_data = merge_data(train_data_ads_df, train_data_feeds_df)
    test_data = merge_data(test_data_df, train_data_feeds_df)
    # GC
    del train_data_feeds_df, train_data_ads_df, test_data_df
    gc.collect()

    # process
    sp, d, s = 1, 1, 1
    names = []
    for c in train_data.columns:
        if c in SPARSE_NAMES:
            names.append(f'C{sp}')
            sp += 1
        elif c in DENSE_NAMES:
            names.append(f'I{d}')
            d += 1
        elif c in SEQ_NAMES:
            names.append(f'S{s}')
            s += 1
        else:
            names.append(c)
    train_data.columns = names
    names.remove('label')
    test_data.columns = names

    sp = [n for n in names if n[0] == 'C']
    d = [n for n in names if n[0] == 'I']
    s = [n for n in names if n[0] == 'S']

    # Map multi-value attribution to numerical order
    def map_seq_func(x):
        if x.name in s:
            # print(x.name)

            def map_item(item):
                if not isinstance(item, str):
                    return item
                res = ''
                for v in item.split(SEQ_SPLIT):
                    if v not in seq_map[x.name]:
                        seq_map[x.name][v] = len(seq_map[x.name])
                    n = seq_map[x.name][v]
                    res = f'{n}' if res == '' else f'{res},{n}'
                return res

            seq_map.setdefault(x.name, {})
            x = x.apply(map_item)
        return x

    logger.info("Start to process sequence data...")
    train_data = train_data.apply(map_seq_func)
    test_data = test_data.apply(map_seq_func)

    logger.info(f'train_data NAN rate: {train_data.isna().sum().sum() / len(train_data) / len(train_data.columns):.3f}')
    logger.info(f'test_data NAN rate: {test_data.isna().sum().sum() / len(test_data) / len(test_data.columns):.3f}')
    logger.info("Start to generate feature columns...")
    train_data = base.process(train_data, sp, d, s)
    test_data = base.process(test_data, sp, d, s)

    fc = base.gen_feature_columns(train_data, sp, d, s, seq_map)
    fc, train_data = base.split_dataset(train_data, fc, 0)
    fc, test_data = base.split_dataset(test_data, fc, 0)
    return fc, train_data, test_data


def merge_data(ads_df, feeds_df):
    """
    Minor modifications from baseline code.

    :param input_dir:
    :param ads_df:
    :param feeds_df:
    :param file_name:
    :return:
    """
    # ! maybe we can ignore `date_flg` column
    right = ['u_userId']
    train_merge_df = pd.merge(ads_df, feeds_df, left_on=['user_id'], right_on=right, how='left')
    train_merge_df = train_merge_df.drop(columns=right)
    for n in DENSE_NAMES:
        if n in train_merge_df.columns:
            train_merge_df[n] = train_merge_df[n].values.astype('float')
    # train_merge_df.to_csv(path_or_buf=os.path.join(input_dir, file_name), encoding="utf_8_sig", index=False)
    return train_merge_df
