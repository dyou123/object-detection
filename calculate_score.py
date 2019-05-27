# 图像培训平台 框的匹配,得分，偏移量，属性是否正确对应
# marker.json为人工标注结果，answer.json为答案
# data_rect的基本形式为(x,y,w,h),坐标系左上角为0，向下为y，向右为x
import json
import numpy as np

with open('answer.json', 'r',encoding='utf-8') as f:
    data_answer = json.load(f)
with open('marker.json', 'r',encoding='utf-8') as f:
    data_marker = json.load(f)

def calculate_score(x1, y1, w1, h1, x2, y2, w2, h2):
    # 分别判断x与y两个维度，x维度包括：左左，左中，左右，中中，中右，右右
    sw = 0
    if x1 + w1 <= x2:  # ('左左')
        sw = 0
    elif x1 <= x2 and x1 + w1 >= x2 and x1 + w1 <= x2 + w2:  # ('左中')
        sw = x1 + w1 - x2
    elif x1 <= x2 and x1 + w1 >= x2 + w2:  # ('左右')
        sw = w2
    elif x1 >= x2 and x1 + w1 <= x2 + w2:  # ('中中')
        sw = w1
    elif x1 >= x2 and x1 <= x2 + w2 and x1 + w1 >= x2 + w2:  # ('中右')
        sw = x2 + w2 - x1
    elif x1 >= x2 + w2:  # ('右右')
        sw = 0
    else:  # ('other types')
        sw = 0

    # 分别判断x与y两个维度，y维度包括：上上，上中，上下，中中，中下，下下
    sh = 0
    if y1 + h1 <= y2:  # ('上上')
        sh = 0
    elif y1 <= y2 and y1 + h1 >= y2 and y1 + h1 <= y2 + h2:  # ('上中')
        sh = y1 + h1 - y2
    elif y1 <= y2 and y1 + y1 >= y2 + y2:  # ('上下')
        sh = h2
    elif y1 >= y2 and y1 + h1 <= y2 + h2:  # ('中中')
        sh = h1
    elif y1 >= y2 and y1 <= y2 + h2 and y1 + h1 >= y2 + h2:  # ('中下')
        sh = y2 + h2 - y1
    elif y1 >= y2 + h2:  # ('下下')
        sh = 0
    else:  # ('other types')
        sh = 0

    S = sw * sh
    score = S / (w1 * h1 + w2 * h2 - S)
    return score

def main(data_answer, data_marker,  bias_a):

    all_boxes,good_boxes,ave_bias,max_bias,min_bias = 0,0,0,0,0

    # bias表示一个包内32张图片包含的7个框的上下左右四个偏移量
    bias = []

    for img_num in range(len(data_answer)):

        # 找到对应name相同的data_marker的img_index位置
        name_list = [i['pic_name'] for i in data_marker]
        img_index = name_list.index(data_answer[img_num]['pic_name'])

        # 判断data_marker[img_index]['valid'] == 'valid'
        if data_marker[img_index]['valid'] == 'valid':

            # 每个答案框，都要对应一个标注框；若对应score小于0.2，则视为没有对应上；算不通过，但是不计入bias
            for box_num in range(len(data_answer[img_num]['framedata'])):
                x2 = data_answer[img_num]['framedata'][box_num]['x']
                y2 = data_answer[img_num]['framedata'][box_num]['y']
                w2 = data_answer[img_num]['framedata'][box_num]['w']
                h2 = data_answer[img_num]['framedata'][box_num]['h']

                # each_score目的是找到对应的那个框
                each_score = [0] * len(data_marker[img_index]['framedata'])

                for com_box in range(len(data_marker[img_index]['framedata'])):

                    x1 = data_marker[img_index]['framedata'][com_box]['x']
                    y1 = data_marker[img_index]['framedata'][com_box]['y']
                    w1 = data_marker[img_index]['framedata'][com_box]['w']
                    h1 = data_marker[img_index]['framedata'][com_box]['h']

                    # 处理w和h为负数的情况,把起始点移到左上角
                    if w1 < 0: x1 = x1 + w1; w1 = - w1
                    if h1 < 0: y1 = y1 + h1; h1 = - h1
                    if w2 < 0: x2 = x2 + w2; w2 = - w2
                    if h2 < 0: y2 = y2 + h2; h2 = - h2

                    each_score[com_box] = calculate_score(x1, y1, w1, h1, x2, y2, w2, h2)
                    # print(each_score)

                max_index = each_score.index(max(each_score))

                # 判断max_index对应的score是否大于0.2，是的话，可以加入bias的序列
                if each_score[max_index] > 0.2:
                    all_boxes += 1
                    # 确定对应标注框data_marker[img_index]['framedata'][max_index]后，就可以加入bias中
                    zuo_bias = abs(data_marker[img_index]['framedata'][max_index]['x'] - x2)
                    you_bias = abs(data_marker[img_index]['framedata'][max_index]['x'] + data_marker[img_index]['framedata'][max_index]['w'] - x2 - w2)
                    sha_bias = abs(data_marker[img_index]['framedata'][max_index]['y'] - y2)
                    xia_bias = abs(data_marker[img_index]['framedata'][max_index]['y'] + data_marker[img_index]['framedata'][max_index]['h'] - y2 - h2)
                    bias.append(zuo_bias)
                    bias.append(you_bias)
                    bias.append(sha_bias)
                    bias.append(xia_bias)

                    # 首先判断属性是否对应（val1,val2,val3数目不固定）
                    shuxing, val_index = 1, 4
                    while val_index < len(data_answer[img_num]['framedata'][box_num]):
                        if data_answer[img_num]['framedata'][box_num]['val%s' % (val_index - 3)] == \
                                data_marker[img_index]['framedata'][max_index]['val%s' % (val_index - 3)]:
                            val_index += 1
                        else:
                            shuxing = 0
                            break
                    # 判断该框是否合格；属性对应 + bias小于设定值 + score大于0.2
                    if shuxing and max(zuo_bias,you_bias,sha_bias,xia_bias) <= bias_a:
                        good_boxes += 1

    max_bias = max(bias)
    min_bias = min(bias)
    ave_bias =np.mean(bias)

    return all_boxes, good_boxes, ave_bias, max_bias, min_bias

if __name__ == '__main__':
    all_boxes, good_boxes, ave_bias, max_bias, min_bias = main(data_answer, data_marker, 5)
    print(all_boxes, good_boxes, ave_bias, max_bias, min_bias)


