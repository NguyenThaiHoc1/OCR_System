import json


def read_config(path):
    """
    This function to read configurate of program
    :param path:
    :return:
    """
    f = open(path, 'r')
    data = json.load(f)
    f.close()
    return data


def convert_tuple_from_string(stirng_tuple):
    return tuple(map(int, stirng_tuple.split(', ')))
