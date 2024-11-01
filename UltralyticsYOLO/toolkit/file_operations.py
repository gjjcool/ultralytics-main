import os
import shutil


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)  # 返回当前目录下的所有文件及文件夹的列表
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):  # 判断是否为文件
            os.remove(file_path)
        elif os.path.isdir(file_path):  # 判断是否为目录
            shutil.rmtree(file_path)  # 递归的删除文件


def clear_or_new_dir(dir):
    if os.path.exists(dir):
        del_file(dir)
    else:
        os.makedirs(dir)


def new_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# 遍历 base_path 文件夹下的所有文件并返回文件路径和文件名
def find_all_file_paths(base_path):
    res = []
    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            res.append((file_path, str(file)))
    return res


def get_dir_name(file_path):
    return os.path.dirname(file_path)


def get_suffix(file_path):
    return os.path.splitext(file_path)[-1].lower()


def get_stem(file_path):
    return os.path.basename(file_path).split('.')[0]
