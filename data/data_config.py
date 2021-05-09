

def get_dataset_info(dataset_type):
    """定义数据集名称及其数据根"""
    root = ''
    if dataset_type == 'LEVIR_CD':
        root = 'path-to-LEVIR_CD-dataroot'
    # add more dataset ...
    else:
        raise TypeError("not define the %s" % dataset_type)

    return root
