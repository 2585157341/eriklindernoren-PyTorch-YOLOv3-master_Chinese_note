

def parse_model_config(path):
    """
    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）
    
    """
    # 加载文件并过滤掉文本中多余内容
    file = open(path, 'r')
    lines = file.read().split('\n')                 # store the lines in a list等价于readlines
    lines = [x for x in lines if x and not x.startswith('#')]    # 去掉空行以及去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    # cfg文件中的每个块用[]括起来最后组成一个列表，一个block存储一个块的内容，即每个层用一个字典block存储。
    module_defs = []
    for line in lines:
        if line.startswith('['):  # 这是cfg文件中一个层(块)的开始
            module_defs.append({}) # 这个块（字典）加入到blocks列表中去
            module_defs[-1]['type'] = line[1:-1].rstrip()  # 把cfg的[]中的块名作为键type的值
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")  #按等号分割
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()#左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
