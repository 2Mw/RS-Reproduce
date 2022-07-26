import cf.preprocess.data as base
import numpy as np
import os


"""
Deprecated temporarily: Use baseline model methods
"""

# Competition Link: https://developer.huawei.com/consumer/cn/activity/starAI2022/algo/competition.html
# dataset link: https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/2022/AI/238143c8f2b211ec84305c80b6c7dac5/2022_3_data.zip

# ===================   Target domain  train_data_ads.csv =========================

# 35 items
ADS_NAMES = [
    # 日志id，是否点击，    用户id，   年龄， 性别，     常住地-省，  常住地-城市，常住地-等级，    设备序列
    'log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'city', 'city_rank', 'series_dev',
    # 序列组，          emui 设备，  设备名称      设备尺寸        网络类型        任务id    广告 id      广告生成类型
    'series_group', 'emui_dev', 'device_name', 'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd',
    # 广告优先级 id    交互类型          广告位置    投放的网站   投放的app          hispace应用标签
    'adv_prim_id', 'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id', 'hispace_app_tags',
    # 投放app所属第二类别   app评分       广告点击列表1(历史)       广告点击列表2         广告点击列表3
    'app_second_class', 'app_score', 'ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003',
    # 广告关闭列表1           广告关闭列表2             广告关闭列表3         xx日期    信息流分类偏好
    'ad_close_list_v001', 'ad_close_list_v002', 'ad_close_list_v003', 'pt_d', 'u_newsCatInterestsST',
    # 日均有效刷新次数    用户活跃度
    'u_refreshTimes', 'u_feedLifeCycle']

# Multi-value attributions:
#   ad_click_list_v001      25
#   ad_click_list_v002      26
#   ad_click_list_v003      27
#   ad_close_list_v001      28
#   ad_close_list_v002      29
#   ad_close_list_v003      30
#   u_newsCatInterestsST    32

# removed ID columns: log_id(0), pt_d(31)


# ===================   Source domain  train_data_feeds.csv =========================

# 28 items
FEEDS_NAMES = [
    # 用户 ID     用户手机价格      浏览器用户活跃度        业务类型            活跃度             有效刷新次数
    'u_userId', 'u_phonePrice', 'u_browserLifeCycle', 'u_browserMode', 'u_feedLifeCycle', 'u_refreshTimes',
    # 信息分类偏好            信息分类讨厌          信息类别感兴趣             用户点击信息类型       广告文档id
    'u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news', 'i_docId',
    # 广告来源id        广告区域实体      广告类型    广告实体        广告不喜欢的次数    广告刷新次数     广告数据类型
    'i_s_sourceId', 'i_regionEntity', 'i_cat', 'i_entities', 'i_dislikeTimes', 'i_upTimes', 'i_dtype',
    # entity- -     -       -       -       实体分类      end-time  标签     **标签     专业程度
    'e_ch', 'e_m', 'e_po', 'e_pl', 'e_rn', 'e_section', 'e_et', 'label', 'cillabel', 'pro']

# Multi-value attributions
#   u_newsCatInterests      6
#   u_newsCatDislike        7
#   u_newsCatInterestsST    8
#   u_click_ca2_news        9
#   i_entities              14

# remove columns: e_et(24)

# Overlapped Attributions:  key:userId, rename: u_newsCatInterestsST, store one: (u_refreshTimes, u_feedLifeCycle)


# ===================   Aggregation Attributions(all)  =========================

# Aggregate by userId (56 columns)
# Source domain rename: u_newsCatInterestsST -> u_newsCatInterestsST_2,  label -> label_2

ALL_NAMES = [
    # 'C1', 'I1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'
    'user_id', 'age', 'gender', 'residence', 'city', 'city_rank', 'series_dev', 'series_group', 'emui_dev',
    # 'C9', 'I2', 'C10', 'C11', 'C12', 'C13', 'C14'
    'device_name', 'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id',
    # 'C15', 'C16', 'C17', 'C18', 'C19', 'C20'
    'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id', 'hispace_app_tags', 'app_second_class',
    # 'I3', 'C21', 'C22', 'C23', 'C24'
    'app_score', 'ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003', 'ad_close_list_v001',
    # 'C25', 'C26', 'C27', 'I4'
    'ad_close_list_v002', 'ad_close_list_v003', 'u_newsCatInterestsST', 'u_refreshTimes',
    # 'C28', 'I5', 'C29',  'C30', 'C31'
    'u_feedLifeCycle', 'u_phonePrice', 'u_browserLifeCycle', 'u_browserMode', 'u_newsCatInterests',
    # 'C32', 'C33', 'C34', 'C35', 'C36'
    'u_newsCatDislike', 'u_newsCatInterestsST_2', 'u_click_ca2_news', 'i_docId', 'i_s_sourceId',
    # 'C37',        'C38',      'C39',      'I6',               'I7',       'C40',      'C41', 'C42', 'C43'
    'i_regionEntity', 'i_cat', 'i_entities', 'i_dislikeTimes', 'i_upTimes', 'i_dtype', 'e_ch', 'e_m', 'e_po',
    # 'C44', 'C45', 'C46', 'C47', 'C48', 'C49'
    'e_pl', 'e_rn', 'e_section', 'label_2', 'cillabel', 'pro']

# numeric 4 feature, categorical
MAPPED_NAMES = ['C1', 'I1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'I2', 'C10', 'C11', 'C12', 'C13', 'C14',
                'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'I3', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'I4',
                'C28', 'I5', 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'I6', 'I7',
                'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49']

SEP = ','


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeroc_process: str = 'mms'):
    """
    Create HUAWEI dataset.

    :param file:
    :param sample_num:
    :param test_size:
    :param numeroc_process:
    :return:
    """
    if sample_num != -1:
        raise ValueError('Fliggy sample_num must be all(-1).')
    dirname = os.path.dirname(file)
    target_file, src_file = [os.path.join(dirname, i) for i in ['train_data_ads.csv', 'train_data_feeds.csv']]
    target = base.read_raw_data(target_file, sample_num, SEP)
    src = base.read_raw_data(src_file, sample_num, SEP)
    data = []
    target_dict, src_dict = {}, {}
    # record the multi-value attribution's vocabulary size
    target_m_index = [25, 26, 27, 28, 29, 30, 32]
    target_item_dict = {
        'ad_click_list_v001': {},  # 25
        'ad_click_list_v002': {},  # 26
        'ad_click_list_v003': {},  # 27
        'ad_close_list_v001': {},  # 28
        'ad_close_list_v002': {},  # 29
        'ad_close_list_v003': {},  # 30
        'u_newsCatInterestsST': {}  # 32
    }
    src_m_index = [6, 7, 8, 9, 14]
    src_item_dict = {
        'u_newsCatInterests': {},  # 6
        'u_newsCatDislike': {},  # 7
        'u_newsCatInterestsST': {},  # 8
        'u_click_ca2_news': {},  # 9
        'i_entities': {},  # 14
    }
    # Process
    for t in target:
        for i in target_m_index:
            mv = t[i]
            res = ''
            if '^' in mv:  # Check if is multi-value
                mv = mv.split('^')  # split
            else:
                mv = [mv]
            for v in mv:
                if len(v) > 0:
                    # check if exists in map
                    col_name = ADS_NAMES[i]
                    if v not in target_item_dict[col_name]:
                        target_item_dict[col_name][v] = len(target_item_dict[col_name])
                    n = target_item_dict[col_name][v]
                    res = f'{n}' if res == '' else f'{res},{n}'
            t[i] = res
        # remove useless columns: log_id(0), pt_d(31)
        t.pop(31)
        t.pop(0)
        target_dict[t[1]] = t

    # 如何整合源域的信息
    # 源域中用户行为也是多个
    for s in src:
        for i in src_m_index:
            mv = s[i]
            res = ''
            if '^' in mv:  # Check if is multi-value
                mv = mv.split('^')  # split
            else:
                mv = [mv]
            for v in mv:
                if len(v) > 0:
                    # check if exists in map
                    col_name = ADS_NAMES[i]
                    if v not in src_item_dict[col_name]:
                        src_item_dict[col_name][v] = len(src_item_dict[col_name])
                    n = src_item_dict[col_name][v]
                    res = f'{n}' if res == '' else f'{res},{n}'
            s[i] = res
        # remove columns: e_et(24)
        s.pop(24)
        src_dict[s[0]] = s
