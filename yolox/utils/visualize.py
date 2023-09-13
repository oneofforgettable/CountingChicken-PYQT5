#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
import math

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

####################################### 画轨迹 使用投票的方式进行计数 只记录数目 ################################################
def plot_tracking_vote(total_id_top_1, total_id_bottom_1, total_id_left_1, total_id_right_1,
                    total_id_top_2, total_id_bottom_2, total_id_left_2, total_id_right_2,
                    total_id_top_3, total_id_bottom_3, total_id_left_3, total_id_right_3,
                    center_points,direction, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    line_x = [0, 0, 0]
    line_y = [0, 0, 0]

    # 画出每一只鸡，四个方向
    # 从上到下 从下到上，从左到右，从右到左
    count_chicken_top_1 = []
    count_chicken_bottom_1 = []
    count_chicken_left_1 = []
    count_chicken_right_1 = []
    count_chicken_top_2 = []
    count_chicken_bottom_2 = []
    count_chicken_left_2 = []
    count_chicken_right_2 = []
    count_chicken_top_3 = []
    count_chicken_bottom_3 = []
    count_chicken_left_3 = []
    count_chicken_right_3 = []

    line_x[0] = int(int(im_w) * 0.25)
    line_y[0] = int(int(im_h) * 0.25)
    line_x[1] = int(int(im_w) * 0.5)
    line_y[1] = int(int(im_h) * 0.5)
    line_x[2] = int(int(im_w) * 0.75)
    line_y[2] = int(int(im_h) * 0.75)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x_c = int(x1+w/2)
        y_c = int(y1+h/2)

        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        # 使用字典来保存每一只鸡的中心点
        if obj_ids is not None:
            if obj_ids[i] in center_points:
                center_points.get(obj_ids[i]).append(x_c)
                center_points.get(obj_ids[i]).append(y_c)
            else:
                center_points[obj_ids[i]] = [x_c, y_c]
                '''识别小鸡的前进方向'''
                '''从上到下 '''
                if y_c <= 200:
                    direction[0] = direction[0] + 1
                '''从下到上'''
                if y_c >= (im_h - 200):
                    direction[1] = direction[1] + 1
                '''从左到右'''
                if x_c <= 200:
                    direction[2] = direction[2] + 1
                '''从右到左'''
                if x_c >= (im_w-200):
                    direction[3] = direction[3] + 1
            # 根据鸡的中心点画轨迹
            center = center_points.get(obj_ids[i])
            count_center = int(len(center)/2)
            if count_center > 1:
                for k in range(count_center-1):
                    cv2.line(im, (center[k*2], center[k*2+1]), (center[(k+1)*2], center[(k+1)*2+1]), color=color, thickness=line_thickness)
                '''从右到左'''
                if center[0] >= line_x[0] >= center[len(center) - 2]:
                    count_chicken_right_1.append(obj_ids[i])
                if center[0] >= line_x[1] >= center[len(center) - 2]:
                    count_chicken_right_2.append(obj_ids[i])
                if center[0] >= line_x[2] >= center[len(center) - 2]:
                    count_chicken_right_3.append(obj_ids[i])
                '''从左到右'''
                if center[0] <= line_x[0] <= center[len(center) - 2]:
                    count_chicken_left_1.append(obj_ids[i])
                if center[0] <= line_x[1] <= center[len(center) - 2]:
                    count_chicken_left_2.append(obj_ids[i])
                if center[0] <= line_x[2] <= center[len(center) - 2]:
                    count_chicken_left_3.append(obj_ids[i])
                '''从上到下 '''
                if center[1] <= line_y[0] <= center[len(center) - 1]:
                    count_chicken_top_1.append(obj_ids[i])
                if center[1] <= line_y[1] <= center[len(center) - 1]:
                    count_chicken_top_2.append(obj_ids[i])
                if center[1] <= line_y[2] <= center[len(center) - 1]:
                    count_chicken_top_3.append(obj_ids[i])
                '''从下到上'''
                if center[1] >= line_y[0] >= center[len(center) - 1]:
                    count_chicken_bottom_1.append(obj_ids[i])
                if center[1] >= line_y[1] >= center[len(center) - 1]:
                    count_chicken_bottom_2.append(obj_ids[i])
                if center[1] >= line_y[2] >= center[len(center) - 1]:
                    count_chicken_bottom_3.append(obj_ids[i])
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.circle(im, (int(x_c), int(y_c)), 1, color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    ## cout chicken
    total_id_top_1 += count_chicken_top_1
    total_id_top_2 += count_chicken_top_2
    total_id_top_3 += count_chicken_top_3
    total_id_bottom_1 += count_chicken_bottom_1
    total_id_bottom_2 += count_chicken_bottom_2
    total_id_bottom_3 += count_chicken_bottom_3
    total_id_left_1 += count_chicken_left_1
    total_id_left_2 += count_chicken_left_2
    total_id_left_3 += count_chicken_left_3
    total_id_right_1 += count_chicken_right_1
    total_id_right_2 += count_chicken_right_2
    total_id_right_3 += count_chicken_right_3
    total_id_top_1 = list(set(total_id_top_1))
    total_id_top_2 = list(set(total_id_top_2))
    total_id_top_3 = list(set(total_id_top_3))
    total_id_bottom_1 = list(set(total_id_bottom_1))
    total_id_bottom_2 = list(set(total_id_bottom_2))
    total_id_bottom_3 = list(set(total_id_bottom_3))
    total_id_left_1 = list(set(total_id_left_1))
    total_id_left_2 = list(set(total_id_left_2))
    total_id_left_3 = list(set(total_id_left_3))
    total_id_right_1 = list(set(total_id_right_1))
    total_id_right_2 = list(set(total_id_right_2))
    total_id_right_3 = list(set(total_id_right_3))
    count_top = [len(total_id_top_1), len(total_id_top_2), len(total_id_top_3)]
    count_bottom = [len(total_id_bottom_1), len(total_id_bottom_2), len(total_id_bottom_3)]
    count_left = [len(total_id_left_1), len(total_id_left_2), len(total_id_left_3)]
    count_right = [len(total_id_right_1), len(total_id_right_2), len(total_id_right_3)]

    '''
    第一位表示过中间线的鸡的数量，
    第二位表示过三条线的鸡的平均数（A + B + C） / 2
    第三位表示A * 0.25 + B * 0.5 + c * 0.25
    第四位表示取三条线中的最大值
    '''
    chicken_number = [0, 0, 0, 0]

    d = direction.index(max(direction))
    if d == 0:# top
        # print('从上到下')
        chicken_number[0] = count_top[1]
        chicken_number[1] = math.ceil(np.mean(count_top))
        chicken_number[2] = math.ceil(count_top[0] * 0.25 + count_top[1] * 0.5 + count_top[2] * 0.25)
        chicken_number[3] = np.max(count_top)
        cv2.line(im, (1, int(line_y[0])), (int(im_w) - 1, int(line_y[0])), thickness=2, color=(255, 0, 0))
        cv2.line(im, (1, int(line_y[1])), (int(im_w) - 1, int(line_y[1])), thickness=2, color=(0, 255, 0))
        cv2.line(im, (1, int(line_y[2])), (int(im_w) - 1, int(line_y[2])), thickness=2, color=(0, 0, 255))
    elif d == 1:# bottom
        # print('从下到上')
        chicken_number[0] = count_bottom[1]
        chicken_number[1] = math.ceil(np.mean(count_bottom))
        chicken_number[2] = math.ceil(count_bottom[0] * 0.25 + count_bottom[1] * 0.5 + count_bottom[2] * 0.25)
        chicken_number[3] = np.max(count_bottom)
        cv2.line(im, (1, int(line_y[0])), (int(im_w) - 1, int(line_y[0])), thickness=2, color=(255, 0, 0))
        cv2.line(im, (1, int(line_y[1])), (int(im_w) - 1, int(line_y[1])), thickness=2, color=(0, 255, 0))
        cv2.line(im, (1, int(line_y[2])), (int(im_w) - 1, int(line_y[2])), thickness=2, color=(0, 0, 255))
    elif d == 2:# left
        # print('从左到右')
        chicken_number[0] = count_left[1]
        chicken_number[1] = math.ceil(np.mean(count_left))
        chicken_number[2] = math.ceil(count_left[0] * 0.25 + count_left[1] * 0.5 + count_left[2] * 0.25)
        chicken_number[3] = np.max(count_left)
        cv2.line(im, (line_x[0], 1), (line_x[0], int(im_h) - 1), thickness=2, color=(255, 0, 0))
        cv2.line(im, (line_x[1], 1), (line_x[1], int(im_h) - 1), thickness=2, color=(0, 255, 0))
        cv2.line(im, (line_x[2], 1), (line_x[2], int(im_h) - 1), thickness=2, color=(0, 0, 255))
    elif d == 3:# right
        # print('从右到左')
        chicken_number[0] = count_right[1]
        chicken_number[1] = math.ceil(np.mean(count_right))
        chicken_number[2] = math.ceil(count_right[0] * 0.25 + count_right[1] * 0.5 + count_right[2] * 0.25)
        chicken_number[3] = np.max(count_right)
        cv2.line(im, (line_x[0], 1), (line_x[0], int(im_h) - 1), thickness=2, color=(255, 0, 0))
        cv2.line(im, (line_x[1], 1), (line_x[1], int(im_h) - 1), thickness=2, color=(0, 255, 0))
        cv2.line(im, (line_x[2], 1), (line_x[2], int(im_h) - 1), thickness=2, color=(0, 0, 255))

    ## 添加图片的标签
    cv2.putText(im, 'frame: %d total_id: %d current_num: %d middle_num: %d avg_num: %d weight_num: %d max_num: %d'
                % (frame_id, len(center_points.keys()), len(tlwhs), chicken_number[0], chicken_number[1], chicken_number[2], chicken_number[3]),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    return im, total_id_top_1, total_id_bottom_1, total_id_left_1, total_id_right_1,\
        total_id_top_2, total_id_bottom_2, total_id_left_2, total_id_right_2,\
        total_id_top_3, total_id_bottom_3, total_id_left_3, total_id_right_3,\
        center_points, direction
####################################### direction 画轨迹 使用投票的方式进行计数 只记录数目 ################################################
def plot_tracking_vote_direction(total_id_top_1, total_id_bottom_1, total_id_left_1, total_id_right_1,
                    total_id_top_2, total_id_bottom_2, total_id_left_2, total_id_right_2,
                    total_id_top_3, total_id_bottom_3, total_id_left_3, total_id_right_3,
                    center_points,direction, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    line_x = [0, 0, 0]
    line_y = [0, 0, 0]

    # 画出每一只鸡，四个方向
    # 从上到下 从下到上，从左到右，从右到左

    count_chicken_left_1 = []
    count_chicken_right_1 = []
    count_chicken_left_2 = []
    count_chicken_right_2 = []
    count_chicken_left_3 = []
    count_chicken_right_3 = []

    line_x[0] = int(int(im_w) * 0.25)
    line_y[0] = int(int(im_h) * 0.25)
    line_x[1] = int(int(im_w) * 0.5)
    line_y[1] = int(int(im_h) * 0.5)
    line_x[2] = int(int(im_w) * 0.75)
    line_y[2] = int(int(im_h) * 0.75)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x_c = int(x1+w/2)
        y_c = int(y1+h/2)

        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        # 使用字典来保存每一只鸡的中心点
        if obj_ids is not None:
            if obj_ids[i] in center_points:
                center_points.get(obj_ids[i]).append(x_c)
                center_points.get(obj_ids[i]).append(y_c)
            else:
                center_points[obj_ids[i]] = [x_c, y_c]
            # 根据鸡的中心点画轨迹
            center = center_points.get(obj_ids[i])
            count_center = int(len(center)/2)
            if count_center > 1:
                for k in range(count_center-1):
                    cv2.line(im, (center[k*2], center[k*2+1]), (center[(k+1)*2], center[(k+1)*2+1]), color=color, thickness=line_thickness)
                '''从右到左'''
                if center[0] >= line_x[0] >= center[len(center) - 2]:
                    count_chicken_right_1.append(obj_ids[i])
                if center[0] >= line_x[1] >= center[len(center) - 2]:
                    count_chicken_right_2.append(obj_ids[i])
                if center[0] >= line_x[2] >= center[len(center) - 2]:
                    count_chicken_right_3.append(obj_ids[i])
                '''从左到右'''
                if center[0] <= line_x[0] <= center[len(center) - 2]:
                    count_chicken_left_1.append(obj_ids[i])
                if center[0] <= line_x[1] <= center[len(center) - 2]:
                    count_chicken_left_2.append(obj_ids[i])
                if center[0] <= line_x[2] <= center[len(center) - 2]:
                    count_chicken_left_3.append(obj_ids[i])
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.circle(im, (int(x_c), int(y_c)), 1, color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    ## cout chicken
    total_id_left_1 += count_chicken_left_1
    total_id_left_2 += count_chicken_left_2
    total_id_left_3 += count_chicken_left_3
    total_id_right_1 += count_chicken_right_1
    total_id_right_2 += count_chicken_right_2
    total_id_right_3 += count_chicken_right_3
    total_id_left_1 = list(set(total_id_left_1))
    total_id_left_2 = list(set(total_id_left_2))
    total_id_left_3 = list(set(total_id_left_3))
    total_id_right_1 = list(set(total_id_right_1))
    total_id_right_2 = list(set(total_id_right_2))
    total_id_right_3 = list(set(total_id_right_3))
    count_left = [len(total_id_left_1), len(total_id_left_2), len(total_id_left_3)]
    count_right = [len(total_id_right_1), len(total_id_right_2), len(total_id_right_3)]

    '''
    第一位表示过中间线的鸡的数量，
    第二位表示过三条线的鸡的平均数（A + B + C） / 2
    第三位表示A * 0.25 + B * 0.5 + c * 0.25
    第四位表示取三条线中的最大值
    '''
    chicken_number = [0, 0, 0, 0]

    d = direction.index(max(direction))

    if d == 2:# left
        # print('从左到右')
        chicken_number[0] = count_left[1]
        chicken_number[1] = math.ceil(np.mean(count_left))
        chicken_number[2] = math.ceil(count_left[0] * 0.25 + count_left[1] * 0.5 + count_left[2] * 0.25)
        chicken_number[3] = np.max(count_left)
        cv2.line(im, (line_x[0], 1), (line_x[0], int(im_h) - 1), thickness=2, color=(255, 0, 0))
        cv2.line(im, (line_x[1], 1), (line_x[1], int(im_h) - 1), thickness=2, color=(0, 255, 0))
        cv2.line(im, (line_x[2], 1), (line_x[2], int(im_h) - 1), thickness=2, color=(0, 0, 255))
    elif d == 3:# right
        # print('从右到左')
        chicken_number[0] = count_right[1]
        chicken_number[1] = math.ceil(np.mean(count_right))
        chicken_number[2] = math.ceil(count_right[0] * 0.25 + count_right[1] * 0.5 + count_right[2] * 0.25)
        chicken_number[3] = np.max(count_right)
        cv2.line(im, (line_x[0], 1), (line_x[0], int(im_h) - 1), thickness=2, color=(255, 0, 0))
        cv2.line(im, (line_x[1], 1), (line_x[1], int(im_h) - 1), thickness=2, color=(0, 255, 0))
        cv2.line(im, (line_x[2], 1), (line_x[2], int(im_h) - 1), thickness=2, color=(0, 0, 255))

    ## 添加图片的标签
    cv2.putText(im, 'frame: %d total_id: %d current_num: %d middle_num: %d avg_num: %d weight_num: %d max_num: %d'
                % (frame_id, len(center_points.keys()), len(tlwhs), chicken_number[0], chicken_number[1], chicken_number[2], chicken_number[3]),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    return im, total_id_top_1, total_id_bottom_1, total_id_left_1, total_id_right_1,\
        total_id_top_2, total_id_bottom_2, total_id_left_2, total_id_right_2,\
        total_id_top_3, total_id_bottom_3, total_id_left_3, total_id_right_3,\
        center_points, direction

####################################### 画轨迹 使用Wu Tracker跟踪器   只记录中间线数目 ################################################
def plot_wutracking_vote(total_count, center_points, direction, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    line_x = int(int(im_w) * 0.5)
    line_y = int(int(im_h) * 0.5)

    count_chicken = []

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x_c = int(x1+w/2)
        y_c = int(y1+h/2)

        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        # 使用字典来保存每一只鸡的中心点
        if obj_ids is not None:
            if obj_ids[i] in center_points:
                center_points.get(obj_ids[i]).append(x_c)
                center_points.get(obj_ids[i]).append(y_c)
            else:
                center_points[obj_ids[i]] = [x_c, y_c]

            # 根据鸡的中心点画轨迹
            center = center_points.get(obj_ids[i])
            count_center = int(len(center)/2)
            if count_center > 1:
                for k in range(count_center-1):
                    cv2.line(im, (center[k*2], center[k*2+1]), (center[(k+1)*2], center[(k+1)*2+1]), color=color, thickness=line_thickness)
                '''从右到左'''
                if direction == 'right2left':
                    if center[0] >= line_x >= center[len(center) - 2]:
                        count_chicken.append(obj_ids[i])
                '''从左到右'''
                if direction == 'left2right':
                    if center[0] <= line_x <= center[len(center) - 2]:
                        count_chicken.append(obj_ids[i])

        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.circle(im, (int(x_c), int(y_c)), 1, color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    ## cout chicken
    total_count += count_chicken
    total_count = list(set(total_count))

    cv2.line(im, (line_x, 1), (line_x, int(im_h) - 1), thickness=2, color=(0, 255, 0))

    ## 添加图片的标签
    cv2.putText(im, 'frame: %d total_id: %d current_num: %d total_num: %d '
                % (frame_id, len(center_points.keys()), len(tlwhs), len(total_count)),(0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    return im, total_count, center_points


####################################### 使用Wu Tracker跟踪器   只记ID线数目 ################################################
def plot_wutracking_id(total_count, center_points, direction, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))

    count_chicken = []

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh

        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        count_chicken.append(obj_id)
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    ## cout chicken
    total_count += count_chicken
    total_count = list(set(total_count))

    ## 添加图片的标签
    cv2.putText(im, 'frame: %d total_id: %d current_num: %d total_num: %d '
                % (frame_id, len(center_points.keys()), len(tlwhs), len(total_count)),(0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    return im, total_count, center_points

####################################### 使用Wu Tracker跟踪器   只记line线数目 ################################################
def plot_wutracking_line(total_count, center_points, direction, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    line_x = int(int(im_w) * 0.5)
    count_chicken = []
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        if x1 <= line_x <= x1+w :
            count_chicken.append(obj_id)

        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    ## cout chicken
    total_count += count_chicken
    total_count = list(set(total_count))

    cv2.line(im, (line_x, 1), (line_x, int(im_h) - 1), thickness=2, color=(0, 255, 0))

    ## 添加图片的标签
    cv2.putText(im, 'frame: %d total_id: %d current_num: %d total_num: %d '
                % (frame_id, len(center_points.keys()), len(tlwhs), len(total_count)),(0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    return im, total_count, center_points

####################################### 使用Wu Tracker跟踪器   只记region线数目 ################################################
def plot_wutracking_region(total_count, center_points, direction, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    line_x_left = int(int(im_w) * 0.5)-100
    line_x_right = int(int(im_w) * 0.5)+100

    count_chicken = []

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x_c = int(x1+w/2)
        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        if line_x_left <= x_c <= line_x_right:
            count_chicken.append(obj_id)
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    ## cout chicken
    total_count += count_chicken
    total_count = list(set(total_count))
    cv2.line(im, (line_x_left, 1), (line_x_left, int(im_h) - 1), thickness=2, color=(0, 255, 0))
    cv2.line(im, (line_x_right, 1), (line_x_right, int(im_h) - 1), thickness=2, color=(0, 255, 0))
    ## 添加图片的标签
    cv2.putText(im, 'frame: %d total_id: %d current_num: %d total_num: %d '
                % (frame_id, len(center_points.keys()), len(tlwhs), len(total_count)),(0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    return im, total_count, center_points



_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
