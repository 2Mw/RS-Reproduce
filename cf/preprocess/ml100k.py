import copy
import os
import numpy as np
import pandas as pd

import cf.preprocess.data as base
from cf.utils.logger import logger
from cf.preprocess.feature_column import *

# user
user_col = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
# ratings
ratings_col = ['user_id', 'movie_id', 'rating', 'timestamp']
# movies
movies_col = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']
genres_col = ['genre_unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
              'Western']
movies_col = movies_col + genres_col

SEP1, SEP2 = '|', '\t'


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    """
    create Movielens-100k dateset for double tower (recall model)

    :param file: the filepath of u.data of movielen-100k
    :param sample_num:
    :param test_size:
    :param numeric_process:
    :return:
    """
    if file.find('u.data') == -1:
        e = f'Please specify the filepath of `u.data` in the file param: {file}'
        logger.error(e)
        raise FileNotFoundError(e)
    dirname = os.path.dirname(file)
    user_file, movie_file = [os.path.join(dirname, i) for i in ['u.user', 'u.item']]
    rating_file = file
    users = base.read_data(user_file, -1, SEP1, user_col)
    ratings = base.read_data(rating_file, -1, SEP2, ratings_col)
    movies = base.read_data(movie_file, -1, SEP1, movies_col)
    # process user dataset
    base.mapped2sequential(users, ['occupation', 'zip_code'])
    users['gender'] = np.where(users['gender'] == 'M', 0, 1)
    # process movie dataset，必須 +1 防止被 mask
    all_genre = [','.join([str(i+1) for i, x in enumerate(arr) if x == 1]) for arr in movies[genres_col].values]
    movies['all_genres'] = all_genre
    movies['release_date'] = movies['release_date'].fillna('01-Jan-1995')
    movies['release_year'] = movies['release_date'].str[-4:]
    movies['release_year'] = movies['release_year'].fillna(0).astype('int64')
    movies['movie_title'] = movies['movie_title'].str[:-6]
    # 去除无用的列
    movies = movies.drop(columns=['video_release_date', 'IMDb_URL', 'release_date'] + genres_col)
    base.mapped2sequential(movies, ['movie_title'])
    base.std_normalize(movies, ['release_year'])
    # merge three dataset
    ratings_all = ratings.merge(movies, on='movie_id').merge(users, on='user_id')
    ratings_all = ratings_all.sort_values(by=['user_id', 'timestamp'])
    ratings_all['like_type'] = np.where(ratings_all['rating'] >= 3, 'like', 'dislike')
    base.mapped2sequential(ratings_all, ['user_id', 'movie_id'])
    base.std_normalize(ratings_all, ['timestamp'])
    ratings_all = ratings_all.reset_index(drop=True)
    # 将用户看过的所有电影按照喜欢和不喜欢进行分类，作为用户的属性之一
    movie_list_for_user = ratings_all.groupby(['user_id', 'like_type'])['movie_id'].apply(list).reset_index()
    movie_list_for_user = movie_list_for_user.pivot(index='user_id', columns='like_type',
                                                    values='movie_id').reset_index()
    movie_list_for_user.fillna(ratings_all['movie_id'].max() + 1, inplace=True)
    movie_list_for_user['like'] = movie_list_for_user['like'].apply(lambda x: x if type(x) is list else [x])
    movie_list_for_user['dislike'] = movie_list_for_user['dislike'].apply(lambda x: x if type(x) is list else [x])
    # 构造 query 数据集
    user_data = ratings_all[['user_id', 'age', 'gender', 'occupation', 'zip_code']]
    user_data = user_data.drop_duplicates().reset_index(drop=True)
    user_data = user_data.merge(movie_list_for_user)
    # 构造 item 数据集
    movie_data = ratings_all[['movie_id', 'movie_title', 'all_genres', 'release_year']].drop_duplicates().reset_index(
        drop=True)
    movie_data['all_genres'] = movie_data['all_genres'].apply(lambda x: [int(i) for i in x.split(',')])
    # !!! 划分数据集以及为负采样做处理
    lens = user_data.shape[0]
    train_user_data = user_data[:int(lens * 0.9)].reset_index(drop=True)
    test_user_data = user_data[int(lens * 0.9):].reset_index(drop=True)
    # 为每个 train_user 添加正样本
    train_data = None
    for i in train_user_data[['user_id', 'like']].values:
        df = pd.DataFrame([i[0]] * len(i[1]), columns=['user_id'])
        df = df.merge(user_data, on='user_id')
        df['movie_id'] = i[1]
        df = df.merge(movie_data, on='movie_id')
        if train_data is None:
            train_data = df
        else:
            train_data = pd.concat([train_data, df])

    # 象征性地为 test_user_data 添加一个 movie
    test_user_data['movie_id'] = test_user_data['like'].apply(lambda x: x[-1])
    test_user_data = test_user_data.merge(movie_data, on='movie_id')
    # 将数据集顺序打乱
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    # 生成 feature columns
    # origin columns: user_id, age, gender, occupation, zip_code, dislike, like, movie_id, movie_title, all_genres, release_year
    # mapped columns: C1, I1, I2, C2, C3, S1::item, S2::item, C4, C5, S2::genre, I3
    mapped_columns = ['C1::query', 'I1', 'I2', 'C2', 'C3', 'S1::item', 'S2::item', 'C4::item', 'C5', 'S3::genre', 'I3']
    train_data.columns = mapped_columns
    test_user_data.columns = mapped_columns
    fc = [
        SparseFeat('C1::query', user_data['user_id'].max() + 1, np.int32),
        DenseFeat('I1', 1, np.float32),
        DenseFeat('I2', 1, np.float32),
        SparseFeat('C2', user_data['occupation'].max() + 1, np.int32),
        SparseFeat('C3', user_data['zip_code'].max() + 1, np.int32),
        SequenceFeat('S1::item', movie_data['movie_id'].max() + 1, np.int32),
        SequenceFeat('S2::item', movie_data['movie_id'].max() + 1, np.int32),
        SparseFeat('C4::item', movie_data['movie_id'].max() + 1, np.int32),
        SparseFeat('C5', movie_data['movie_title'].max() + 1, np.int32),
        SequenceFeat('S3::genre', 21, np.int64),
        DenseFeat('I3', 1, np.float32),
    ]
    train_x = {f.name: train_data[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
    train_x.update({f.name: train_data[f.name].values for f in fc if isinstance(f, SequenceFeat)})
    query_data = {f.name: test_user_data[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
    query_data.update({f.name: test_user_data[f.name].values for f in fc if isinstance(f, SequenceFeat)})
    # 对于 item 数据进行处理
    item_df = pd.DataFrame([user_data['user_id'][0]] * len(movie_data), columns=['user_id'])
    item_df = item_df.merge(user_data, on='user_id')
    item_df = pd.concat([item_df, movie_data], axis=1)
    item_df.columns = mapped_columns
    item_data = {f.name: item_df[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
    item_data.update({f.name: item_df[f.name].values for f in fc if isinstance(f, SequenceFeat)})
    return fc, train_x, query_data, item_data


def create_dataset_dep(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    """
    @deprecated

    Create Movielens-100k for recall model

    :param file: the filepath of u.data of movielen-100k
    :param sample_num:
    :param test_size:
    :param numeric_process:
    :return:
    """
    if file.find('u.data') == -1:
        e = f'Please specify the filepath of `u.data` in the file param: {file}'
        logger.error(e)
        raise FileNotFoundError(e)
    # raise DeprecationWarning('This method has deprecated')
    dirname = os.path.dirname(file)
    user_file, movie_file = [os.path.join(dirname, i) for i in ['u.user', 'u.item']]
    rating_file = file
    users = base.read_data(user_file, -1, SEP1, user_col)
    ratings = base.read_data(rating_file, -1, SEP2, ratings_col)
    movies = base.read_data(movie_file, -1, SEP1, movies_col)
    # 在 u.item 电影特征中，所属题材是采用 one-hot 的形式进行存储的，将其处理成为多值属性
    all_genre = [','.join([str(i) for i, x in enumerate(arr) if x == 1]) for arr in movies[genres_col].values]
    movies['all_genre'] = all_genre
    # 将 ratings, movies, users 全部聚合在一起
    ratings_all = ratings.merge(movies, on='movie_id').merge(users, on='user_id')
    # 扔掉 one-hot 列，只保留多值属性
    ratings_all = ratings_all.drop(columns=genres_col)
    # 根据 rating 数值来判断用户喜欢还是不喜欢电影，大于 3 为喜欢
    ratings_all['like_type'] = np.where(ratings_all['rating'] >= 3, 'like', 'dislike')
    # 去除后面的年份信息
    ratings_all['movie_name'] = ratings_all['movie_title'].str[:-6]
    # 将 rating 数据根据 user_id 来进行排序，并且使用 timestamp 进行排序来防止出现特征穿越的问题
    ratings_all = ratings_all.sort_values(by=['user_id', 'timestamp'])
    # 将 user_id, movie_id, movie_name 全部转化为连续的 id
    user_ids = ratings_all['user_id'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    movie_ids = ratings_all["movie_id"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    title_ids = ratings_all["movie_name"].unique().tolist()
    title2title_encoded = {x: i for i, x in enumerate(title_ids)}

    occupation_ids = ratings_all['occupation'].unique().tolist()
    oc2oc_encoded = {x: i for i, x in enumerate(occupation_ids)}

    ratings_all['user'] = ratings_all['user_id'].map(user2user_encoded)
    ratings_all['movie'] = ratings_all['movie_id'].map(movie2movie_encoded)
    ratings_all['title_d'] = ratings_all['movie_name'].map(title2title_encoded)
    ratings_all['occupation'] = ratings_all['occupation'].map(oc2oc_encoded)

    # 开始产生训练数据的原型
    # 1. 对每个用户看过的所有电影按照喜欢和不喜欢进行分类
    movie_list = ratings_all.groupby(['user', 'like_type'])['movie'].apply(list).reset_index()
    # 2. 产生每个用户看过的所有电影名称
    title_list = ratings_all.groupby(['user'])['title_d'].apply(list).reset_index()
    # 3. 对每个用户生成其看过所有电影的题材
    genre_list = ratings_all.groupby(['user'])['all_genre'].unique().apply(list).reset_index()
    genre_list['all_genre'] = genre_list['all_genre'].apply(
        lambda x: [int(i) for i in list(set(','.join(x))) if i.isdigit()])  # 对题材进行去重

    # 创建以用户id为行，喜欢类型为列（即正样本和负样本），对应的表格值为电影
    user_video_list = movie_list.pivot(index='user', columns='like_type', values='movie').reset_index()
    user_video_list.fillna(ratings_all['movie'].max() + 1, inplace=True)
    # 生成用户信息
    user_data = ratings_all[['user', 'occupation', 'gender', 'age']]
    # 相当于复制一份数据？
    user_data = user_data.drop_duplicates()
    user_data = user_data.reset_index()
    user_data = user_data.drop('index', axis=1)

    # 生成数据集
    dataset = user_video_list.merge(title_list, on='user').merge(genre_list).merge(user_data)
    dataset['like'] = dataset['like'].apply(lambda x: x if type(x) is list else [x])
    dataset['dislike'] = dataset['dislike'].apply(lambda x: x if type(x) is list else [x])
    # 将最后一个喜欢的电影作为预测标签
    dataset['label'] = dataset['like'].apply(lambda x: x[-1])
    dataset['like'] = dataset['like'].apply(lambda x: x[:-1])
    # 数据集所包含的字段 user, dislike[list], like[list], title_d[list], all_genre[list], occupations, gender, age, label
    # 对于 dense features normalize to [0,1] (in YoutubeDNN chapt3.3)
    dataset['age'] = (dataset['age'] - dataset['age'].min()) / (dataset['age'].max() - dataset['age'].min())
    dataset['gender'] = np.where(dataset['gender'] == 'M', 0, 1)
    mapped_columns = ['user', 'S1::item', 'S2::item', 'S3::item', 'S4::genre', 'C1', 'I1', 'I2', 'label']
    dataset.columns = mapped_columns

    # !important pad_sequence
    # for c in mapped_columns:
    #     if c[0] == 'S':
    #         dataset[c] = ps(dataset[c]).tolist()

    item_class, genre_class = 1684, 21
    seq_map = {'S1::item': item_class, 'S2::item': item_class, 'S3::item': item_class, 'S4::genre': genre_class}

    fc = base.gen_feature_columns(dataset, ['C1'], ['I1', 'I2'], ['S1::item', 'S2::item', 'S3::item', 'S4::genre'],
                                  seq_map)

    return base.split_dataset(dataset, fc, test_size)
