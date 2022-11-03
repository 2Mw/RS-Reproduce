import pandas as pd

import cf.preprocess.data as base
import os

# Columns info

# ========= user_profile.csv =========
# (1) userid：脱敏过的用户ID；
# (2) cms_segid：微群ID；
# (3) cms_group_id：cms_group_id；
# (4) final_gender_code：性别 1:男,2:女；
# (5) age_level：年龄层次；
# (6) pvalue_level：消费档次，1:低档，2:中档，3:高档；
# (7) shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
# (8) occupation：是否大学生 ，1:是,0:否
# (9) new_user_class_level：城市层级
USERS_NAMES = ['userid', 'cms_segid', 'cms_group_id', 'gender', 'age_level', 'pvalue_level', 'shopping_level',
               'occupation', 'city_level']

# ========= ad_feature.csv =============
# (1) adgroup_id：脱敏过的广告ID；
# (2) cate_id：脱敏过的商品类目ID；
# (3) campaign_id：脱敏过的广告计划ID；
# (4) customer_id:脱敏过的广告主ID；
# (5) brand：脱敏过的品牌ID；
# (6) price: 宝贝的价格
AD_NAMES = ['adgroup_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'price']

# ========= raw_sample.csv =============
# (1) user_id：脱敏过的用户ID；
# (2) adgroup_id：脱敏过的广告单元ID；
# (3) time_stamp：时间戳；
# (4) pid：资源位；
# (5) noclk：为1代表没有点击；为0代表点击；
# (6) clk：为0代表没有点击；为1代表点击；
# 我们用前面7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513）。
RAW_NAMES = ['userid', 'adgroupid', 'time_stamp', 'pid', 'noclk', 'clk']

# ========= 整合所有信息 =======
# [ <User feature>, <Ad feature>, label ]
ALL_NAMES = ['userid', 'cms_segid', 'cms_group_id', 'gender', 'age_level', 'pvalue_level', 'shopping_level',
             'occupation', 'city_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'price', 'pid',
             'clk']
MAPPED_NAMES = ['C1', 'C2', 'C3', 'I1', 'C4', 'C5', 'C6',
                'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'I2', 'C14', 'label']
sparse_feature = [f'C{i}' for i in range(1, 15)]
dense_feature = [f'I{i}' for i in range(1, 3)]

# Other
SEP = ','


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    """
    Create TaoBao Ad click dataset

    :param file: file name
    :param sample_num: sample number, -1 means all.
    :param test_size: test size
    :param numeric_process: numeric_process: The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :return: fc, (train_x, train_y), (test_x, test_y), train_x: {'C1': [1,2,3]}
    """
    if sample_num != -1:
        raise ValueError('TaoBao Adclick sample_num must be all(-1).')
    dirname = os.path.dirname(file)
    user_file, ad_file = [os.path.join(dirname, i) for i in ['user_profile.csv', 'ad_feature.csv']]
    clicked = base.read_raw_data(file, sample_num, SEP)
    users = base.read_raw_data(user_file, sample_num, SEP)
    ads = base.read_raw_data(ad_file, sample_num, SEP)
    # drop first line
    clicked, users, ads = clicked[1:], users[1:], ads[1:]
    data = []
    user_dict, ad_dict = {}, {}
    for u in users:
        # gender
        u[3] = 1 if u[3] == '1' else 0
        user_dict[u[0]] = u
    for ad in ads:
        ad[-1] = float(ad[-1])
        ad_dict[ad[0]] = ad
    for r in clicked:
        item = []
        if user_dict.get(r[0]) is None or ad_dict.get(r[2]) is None:
            continue
        item.extend(user_dict[r[0]])
        item.extend(ad_dict[r[2]])
        item.append(r[3])
        item.append(r[-1])
        data.append(item)

    df = pd.DataFrame(data, columns=MAPPED_NAMES)

    df = base.process(df, sparse_feature, dense_feature, numeric_process=numeric_process)

    fc = base.gen_feature_columns(df, sparse_feature, dense_feature)

    return base.split_dataset(df, fc, test_size)
