'''
算法题：判断字符串中{}是否成对匹配，{}匹配，{}{}匹配，{{}不匹配，{}}不匹配，}{不匹配，{{}}匹配，{{}{}}不匹配
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

def checkMatch(str):
    if len(str) == 0:
        return True
    else:
        over = 0
        pos = ''
        index = 0
        while index < len(str):
            if str[index] == '{':
                over += 1
                if pos == 'right':
                    return False
                else:
                    pos = ''
            elif str[index] == '}':
                over -= 1
                if over > 0:
                    pos = 'right'
                elif over == 0:
                    pos = 'left'
                else:
                    return False
            index += 1
    if over == 0:
        return True
    else:
        return False


if __name__ == '__main__':
    right1 = '{}'
    right2 = '{{}}'
    right3 = '{}{{}}{}'
    wrong1 = '{'
    wrong2 = '}'
    wrong3 = '}{'
    wrong4 = '{{}'
    wrong5 = '{}}'
    wrong6 = '{{}{}}'
    print(checkMatch(right1))
    print(checkMatch(right2))
    print(checkMatch(right3))
    print(checkMatch(wrong1))
    print(checkMatch(wrong2))
    print(checkMatch(wrong3))
    print(checkMatch(wrong4))
    print(checkMatch(wrong5))
    print(checkMatch(wrong6))
