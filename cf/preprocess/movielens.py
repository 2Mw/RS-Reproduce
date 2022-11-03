import copy
import os
import pandas as pd

import cf.preprocess.data as base

# NAMES of different file
RATING_NAMES = ['UserID', 'MovieID', 'Rating', 'Timestamp']
USERS_NAMES = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
MOVIES_NAMES = ['MovieID', 'Title', 'Genres']
# Processed names [ <User features>, <Movie features>, Rating ]
# Genres
genres = ['Action', 'Adventure', 'Animation', "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
ALL_NAMES = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code', 'MovieID', 'Genres', 'Rating']
MAPPED_NAMES = ['C1', 'I1', 'I2', 'C2', 'C3', 'C4', 'C5', 'label']
sparse_feature = ['C1', 'C2', 'C3', 'C4', 'C5']
dense_feature = ['I1', 'I2']
# Rating < 3 = 0, Rating > 3 = 1, remove rating == 3.

# Other params
SEP = '::'


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    """
    Create movielens dataset.

    :param file: train file path
    :param sample_num: sample number
    :param test_size: test size
    :param numeric_process: The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :return:fc, (train_x, train_y), (test_x, test_y), train_x: {'C1': [1,2,3]}
    """
    dirname = os.path.dirname(file)
    user_file, movie_file = [os.path.join(dirname, i) for i in ['users.dat', 'movies.dat']]
    rating = base.read_raw_data(file, sample_num, SEP)
    users = base.read_raw_data(user_file, sample_num, SEP)
    movies = base.read_raw_data(movie_file, sample_num, SEP)
    data = []
    # get dict
    users_dict = {}
    movies_dict = {}
    for u in users:
        u[1] = 0 if u[1] == 'F' else 1
        users_dict[u[0]] = u
    for m in movies:
        g = m[2].split('|')
        # 010101110 使用二进制表示不同 genre 的组合
        arr = ['1' if i in g else '0' for i in genres]
        movies_dict[m[0]] = [m[0], int("".join(arr), base=2)]
    for r in rating:
        rate = int(r[-2])
        if rate == 3:  # filter
            continue
        rate = 0 if rate < 3 else 1  # process rate
        item = copy.deepcopy(users_dict[r[0]])
        item.extend(movies_dict[r[1]])
        item.append(rate)
        data.append(item)
    df = pd.DataFrame(data, columns=MAPPED_NAMES)

    df = base.process(df, sparse_feature, dense_feature, numeric_process=numeric_process)

    fc = base.gen_feature_columns(df, sparse_feature, dense_feature)

    return base.split_dataset(df, fc, test_size)
