from numpy import *

# 要实现通过四维向量索引所有的点对
if __name__ == '__main__':
    # n个点

    # 构建哈系表
    hash_table = {}

    # all_key = {'123', '321'}

    # 待添加的一组 key，value
    key_temp = [1, 1, 1, 1]
    key_temp = str(key_temp)
    value_temp = [1, 2]

    # for key in all_key:
    if key_temp in hash_table.keys():
        print('已经存在')
        hash_table[key].append(value_temp)
    else:
        print('尚不存在，需要新建')
        hash_table[key_temp] = value_temp

    print(hash_table)

    # 从哈系表里面取出
    test_key = '[1, 1, 1, 1]'
    if test_key in hash_table.keys():
        test_value = hash_table[test_key]
        print('test_value:', test_value)

    else:
        print('没有这个索引')
