import os

_result_path = "result"


def get_project_path():
    return os.path.dirname(os.path.dirname(__file__))


def check_result():
    dire = os.path.join(os.path.dirname(__file__), _result_path)
    if not os.path.exists(dire):
        os.mkdir(dire)


# 初始化构建存储训练结果的文件夹
check_result()
