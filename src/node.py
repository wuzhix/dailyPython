'''
将数据生成递归结果集
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

arr = (
    {
        'id': 1,
        'name': 'My Documents',
        'pid': 0
    },
    {
        'id': 2,
        'name': 'photos ',
        'pid': 1
    },
    {
        'id': 3,
        'name': 'Friend',
        'pid': 2
    },
    {
        'id': 4,
        'name': 'Wife',
        'pid': 2
    },
    {
        'id': 5,
        'name': 'Company',
        'pid': 2
    },
    {
        'id': 6,
        'name': 'Program Files',
        'pid': 1
    },
    {
        'id': 8,
        'name': 'Java ',
        'pid': 6
    },
)


def get_node(arr, id):
    for node in arr:
        if node['id'] == id:
            if node['pid'] == 0:
                return node['name']
            else:
                return get_node(arr, node['pid']) + '_' + node['name']
    else:
        return ''

for i in arr:
    node = {'id': i['id'], 'name': get_node(arr, i['id'])}
    print(node)
