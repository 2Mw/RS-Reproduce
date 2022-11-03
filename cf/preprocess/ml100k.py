import copy
import os
import numpy as np
import pandas as pd

import cf.preprocess.data as base
from cf.utils.logger import logger

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
    ratings_all['user'] = ratings_all['user_id'].map(user2user_encoded)
    ratings_all['movie'] = ratings_all['movie_id'].map(movie2movie_encoded)
    ratings_all['title_d'] = ratings_all['movie_name'].map(title2title_encoded)

    # 开始产生训练数据的原型
    # 1. 对每个用户看过的所有电影按照喜欢和不喜欢进行分类
    movie_list = ratings_all.groupby(['user', 'like_type'])['movie'].apply(list).reset_index()
    # 2. 产生每个用户看过的所有电影名称
    title_list = ratings_all.groupby(['user'])['title_d'].apply(list).reset_index()
    # 3. 对每个用户生成其看过所有电影的题材
    genre_list = ratings_all.groupby(['user'])['all_genres'].unique().apply(list).reset_index()
    genre_list['all_genres'] = genre_list['all_genres'].apply(
        lambda x: [i for i in list(set(','.join(x))) if i.isdigit()])  # 对题材进行去重

    # 创建以用户id为行，喜欢类型为列（即正样本和负样本），对应的表格值为电影
    user_video_list = movie_list.pivot(index='user', columns='like_type', values='movie').reset_index()
    user_video_list.fillna(ratings_all['movie'].max() + 1, inplace=True)
    # 生成用户信息
    user_data = ratings_all[['user', 'occupation', 'gender']]
    # 相当于复制一份数据？
    user_data = user_data.drop_duplicates()
    user_data = user_data.reset_index()
    user_data = user_data.drop('index', axis=1)

    # 生成数据集
    dataset = user_video_list.merge(title_list, on='user').merge(genre_list).merge(user_data)
    dataset['like'] = dataset['like'].apply(lambda x: x if type(x) is list else [x])
    dataset['dislike'] = dataset['dislike'].apply(lambda x: x if type(x) is list else [x])
    # 将最后一个喜欢的电影作为预测标签
    dataset['predict_labels'] = dataset['like'].apply(lambda x: x[-1])
    dataset['like'] = dataset['like'].apply(lambda x: x[:-1])
    # 数据集所包含的字段 user, dislike[list], like[list], title_d[list], all_genres[list], occupations, gender, predict_labels
    # 对于 dense features normalize to [0,1] (in YoutubeDNN chapt3.3)
