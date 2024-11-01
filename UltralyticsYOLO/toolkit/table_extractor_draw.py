import os
import copy

import cv2
import time
from enum import IntFlag, Enum

import numpy as np


class Direction(IntFlag):
    Up = 1
    Down = 2
    Left = 4
    Right = 8
    Horizontal = Left | Right
    Vertical = Up | Down


class TableType(Enum):
    Empty = 0
    Full = 1
    FullWidth = 2
    Normal = 3


def is_white(bin_img_, pos_):  # 判断二值图像素点是否为白
    return bin_img_[pos_[0], pos_[1]] == 255


def is_black(bin_img_, pos_):  # 判断二值图像素点是否为黑
    return bin_img_[pos_[0], pos_[1]] == 0


def get_table_type(tables_, ft_=10):
    if len(tables_) == 0:
        return TableType.Empty
    for box in tables_:
        if 0 <= box[0][1] <= ft_:
            if 0 <= box[0][0] <= ft_:
                return TableType.Full
            return TableType.FullWidth
    return TableType.Normal


def correct_flipped_box(box_, img_shape_, flip_):
    cor_box_ = copy.deepcopy(box_)
    if flip_ == 0:
        cor_box_[0][0] = img_shape_[0] - box_[0][0]
        cor_box_[1][0] = img_shape_[0] - box_[1][0]
    elif flip_ == 1:
        cor_box_[0][1] = img_shape_[1] - box_[0][1]
        cor_box_[1][1] = img_shape_[1] - box_[1][1]
    elif flip_ == -1:
        cor_box_[0] = [img_shape_[0] - box_[0][0], img_shape_[1] - box_[0][1]]
        cor_box_[1] = [img_shape_[0] - box_[1][0], img_shape_[1] - box_[1][1]]

    tmp_cor_box_ = copy.deepcopy(cor_box_)
    cor_box_[0] = [min(tmp_cor_box_[0][0], tmp_cor_box_[1][0]), min(tmp_cor_box_[0][1], tmp_cor_box_[1][1])]
    cor_box_[1] = [max(tmp_cor_box_[0][0], tmp_cor_box_[1][0]), max(tmp_cor_box_[0][1], tmp_cor_box_[1][1])]
    return cor_box_


def correct_flipped_pos(pos_, img_shape_, flip_):
    cor_pos_ = copy.deepcopy(pos_)
    if flip_ == 0:
        cor_pos_[0] = img_shape_[0] - pos_[0]
    elif flip_ == 1:
        cor_pos_[1] = img_shape_[1] - pos_[1]
    elif flip_ == -1:
        cor_pos_[0] = img_shape_[0] - pos_[0]
        cor_pos_[1] = img_shape_[1] - pos_[1]
    return cor_pos_


def dfs(bin_img_, visited_, x_, y_, connected_set_):
    height_, width_ = bin_img_.shape[:2]
    if x_ < 0 or x_ >= width_ or y_ < 0 or y_ >= height_ or visited_[y_, x_] or is_white(bin_img_, [y_, x_]):
        return

    visited_[y_, x_] = True
    connected_set_.append((y_, x_))

    # 递归访问相邻像素
    dfs(bin_img_, visited_, x_ + 1, y_, connected_set_)
    dfs(bin_img_, visited_, x_ - 1, y_, connected_set_)
    dfs(bin_img_, visited_, x_, y_ + 1, connected_set_)
    dfs(bin_img_, visited_, x_, y_ - 1, connected_set_)


def manhattan_dis(pos1_, pos2_):
    return abs(pos1_[0] - pos2_[0]) + abs(pos1_[1] - pos2_[1])


def calc_bounding_box(connected_set_):
    t_ = b_ = connected_set_[0][0]
    l_ = r_ = connected_set_[0][1]
    for y_, x_ in connected_set_:
        t_ = min(t_, y_)
        b_ = max(b_, y_)
        l_ = min(l_, x_)
        r_ = max(r_, x_)
    return [[t_, l_], [b_, r_]]


class TableExtractor:
    def get_inner_box_normalized(self):
        self.inner_box[0] /= self.H
        self.inner_box[2] /= self.H
        self.inner_box[1] /= self.W
        self.inner_box[3] /= self.W
        return self.inner_box

    def __init__(self, raw_gray_img_, raw_color_img_, min_table_size_=None):
        if min_table_size_ is None:
            min_table_size_ = [100, 100]

        self.__raw_gray_img = raw_gray_img_  # 未经处理的原始图片（灰度图）
        self.__raw_color_img = raw_color_img_  # 用于绘制
        self.__raw_img = copy.deepcopy(raw_color_img_)  # 用于从中截取表格

        # 二值化图片
        _, self.__bin_img = cv2.threshold(raw_gray_img_, 128, 255, cv2.THRESH_BINARY)

        height, width = self.__bin_img.shape[:2]
        self.H, self.W = height, width
        self.__h_mid_pos = width // 2  # 水平中间 x 坐标
        self.__v_mid_pos = height // 2  # 垂直中间 y 坐标

        inner_box = [
            self.__locate_inner_line(range(0, height, 1), Direction.Horizontal),  # top
            self.__locate_inner_line(range(width - 1, -1, -1), Direction.Vertical),  # right
            self.__locate_inner_line(range(height - 1, -1, -1), Direction.Horizontal),  # bottom
            self.__locate_inner_line(range(0, width, 1), Direction.Vertical)  # left
        ]

        self.inner_box = inner_box

        # 内部线框内的图片（不包含两个大框）
        self.__inner_img = self.__bin_img[inner_box[0]:inner_box[2], inner_box[3]:inner_box[1]]
        self.__inner_offset = [inner_box[0], inner_box[3]]  # 内部图基于原图的偏移量（offset_y, offset_x）
        self.__inner_h, self.__inner_w = self.__inner_img.shape[:2]

        self.__brush_color = (255, 0, 0)  # BGR

        self.__min_table_size = min_table_size_

        self.__shrink_times = 3
        target_size = (self.__inner_w // self.__shrink_times, self.__inner_h // self.__shrink_times)
        self.__shrink_inner_img = cv2.resize(self.__inner_img, target_size)
        _, self.__shrink_bin_inner_img = cv2.threshold(self.__shrink_inner_img, 230, 255, cv2.THRESH_BINARY)

        self.__morph_kernel = np.ones((5, 5), np.uint8)
        self.__morph_open_img = cv2.morphologyEx(self.__shrink_bin_inner_img, cv2.MORPH_OPEN, self.__morph_kernel)

        # self.__erode_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 腐蚀核
        self.__erode_kernel = np.ones((3, 3), np.uint8)  # 腐蚀核
        self.__erode_img = cv2.erode(self.__shrink_bin_inner_img, self.__erode_kernel)
        # self.__erode_img = cv2.erode(self.__erode_img, self.__erode_kernel)  # 二次腐蚀

        self.__find_inner_table_base_img = self.__erode_img

    def __draw_solid_box(self, pos_, radius_=1, color_=None, offset_=True, enlarge_=1):
        cp_pos_ = copy.deepcopy(pos_)
        cp_pos_ = [a * enlarge_ for a in cp_pos_]
        if offset_:
            cp_pos_ = [a + b for a, b in zip(cp_pos_, self.__inner_offset)]
        for y_ in range(max(cp_pos_[0] - radius_, 0), min(cp_pos_[0] + radius_ + 1, self.__raw_color_img.shape[0]), 1):
            for x_ in range(max(cp_pos_[1] - radius_, 0), min(cp_pos_[1] + radius_ + 1, self.__raw_color_img.shape[1]), 1):
                self.__raw_color_img[y_, x_] = self.__brush_color if (color_ is None) else color_

    def __draw_line(self, pos1_, pos2_, thickness_=3, color_=(0, 0, 255), offset_=True, enlarge_=1):
        cp_pos1_ = copy.deepcopy(pos1_)
        cp_pos2_ = copy.deepcopy(pos2_)
        cp_pos1_ = [a * enlarge_ for a in cp_pos1_]
        cp_pos2_ = [a * enlarge_ for a in cp_pos2_]
        if offset_:
            cp_pos1_ = [a + b for a, b in zip(cp_pos1_, self.__inner_offset)]
            cp_pos2_ = [a + b for a, b in zip(cp_pos2_, self.__inner_offset)]
        cp_pos1_[0], cp_pos1_[1] = cp_pos1_[1], cp_pos1_[0]
        cp_pos2_[0], cp_pos2_[1] = cp_pos2_[1], cp_pos2_[0]
        cv2.line(self.__raw_color_img, cp_pos1_, cp_pos2_, color_, thickness_)

    def __dfs(self, bin_img_, visited_, x_, y_, connected_set_):
        height_, width_ = bin_img_.shape[:2]
        if x_ < 0 or x_ >= width_ or y_ < 0 or y_ >= height_ or visited_[y_, x_] or is_white(bin_img_, [y_, x_]):
            return

        self.__draw_solid_box([y_, x_], 0, (0, 255, 0))
        visited_[y_, x_] = True
        connected_set_.append((y_, x_))

        # 递归访问相邻像素
        self.__dfs(bin_img_, visited_, x_ + 1, y_, connected_set_)
        self.__dfs(bin_img_, visited_, x_ - 1, y_, connected_set_)
        self.__dfs(bin_img_, visited_, x_, y_ + 1, connected_set_)
        self.__dfs(bin_img_, visited_, x_, y_ - 1, connected_set_)

    def __dfs_on_stack(self, bin_img_, visited_, y_, x_, connected_set_):
        from collections import deque
        height_, width_ = bin_img_.shape[:2]
        stack_ = deque()
        stack_.append([y_, x_])
        while stack_:
            pos_ = stack_.pop()
            y_, x_ = pos_[0], pos_[1]

            self.__draw_solid_box(pos_, 0, (0, 255, 0), enlarge_=self.__shrink_times)
            visited_[y_, x_] = True
            connected_set_.append((y_, x_))

            for next_ in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                next_pos_ = [a + b for a, b in zip(pos_, next_)]
                y_, x_ = next_pos_[0], next_pos_[1]
                if 0 <= y_ < height_ and 0 <= x_ < width_ and (not visited_[y_, x_]) and is_black(bin_img_, next_pos_):
                    stack_.append(next_pos_)

    def __judge_line(self, pos_, dir_):  # 判断当前位置（pos_）是否在水平/竖直（根据 dir_ 方向判断）线上
        offset_list = [0]
        for i in range(8):
            if (1 << i) > ((self.__h_mid_pos if dir_ == Direction.Horizontal else self.__v_mid_pos) - 10):
                break
            offset_list.append(-(1 << i))
            offset_list.append(1 << i)

        for offset in offset_list:
            if dir_ == Direction.Horizontal and self.__bin_img[pos_, self.__h_mid_pos + offset] != 0:
                return False
            if dir_ == Direction.Vertical and self.__bin_img[self.__v_mid_pos + offset, pos_] != 0:
                return False
        return True

    def __judge_boundary(self, pos_, dir_, set_, ft_=3):
        img_ = self.__find_inner_table_base_img
        if dir_ == Direction.Up:
            for y_ in range(pos_[0], pos_[0] - ft_, -1):
                if is_black(img_, [y_, pos_[1]]) and (y_, pos_[1]) in set_:
                    return True
        if dir_ == Direction.Down:
            for y_ in range(pos_[0], pos_[0] + ft_, 1):
                if is_black(img_, [y_, pos_[1]]) and (y_, pos_[1]) in set_:
                    return True
        if dir_ == Direction.Left:
            for x_ in range(pos_[1], pos_[1] - ft_, -1):
                if is_black(img_, [pos_[0], x_]) and (pos_[0], x_) in set_:
                    return True
        if dir_ == Direction.Right:
            for x_ in range(pos_[1], pos_[1] + ft_, 1):
                if is_black(img_, [pos_[0], x_]) and (pos_[0], x_) in set_:
                    return True
        return False

    def __judge_inner_table(self, connected_set_, is_draw_=False):
        bb_ = calc_bounding_box(connected_set_)
        if bb_[1][0] - bb_[0][0] < self.__min_table_size[0] // self.__shrink_times or bb_[1][1] - bb_[0][1] < self.__min_table_size[1] // self.__shrink_times:
            return False  # 若包围和小于阈值，则判为非表格

        fault_tolerance_ = 3
        top_range_, bot_range_, left_range_, right_range_ = ([connected_set_[0], connected_set_[0]],
                                                             [connected_set_[0], connected_set_[0]],
                                                             [connected_set_[0], connected_set_[0]],
                                                             [connected_set_[0], connected_set_[0]])

        # for y_, x_ in connected_set_:
        #     if y_ - fault_tolerance_ <= min(top_range_[0][0], top_range_[1][0]):
        #         if x_ <= top_range_[0][1]:
        #             top_range_[0] = [y_, x_]
        #         elif x_ >= top_range_[1][1]:
        #             top_range_[1] = [y_, x_]
        #
        #     if y_ + fault_tolerance_ >= max(bot_range_[0][0], bot_range_[1][0]):
        #         if x_ <= bot_range_[0][1]:
        #             bot_range_[0] = [y_, x_]
        #         elif x_ >= bot_range_[1][1]:
        #             bot_range_[1] = [y_, x_]
        #
        #     if x_ - fault_tolerance_ <= min(left_range_[0][1], left_range_[1][1]):
        #         if y_ <= left_range_[0][0]:
        #             left_range_[0] = [y_, x_]
        #         elif y_ >= left_range_[1][0]:
        #             left_range_[1] = [y_, x_]
        #
        #     if x_ + fault_tolerance_ >= max(right_range_[0][1], right_range_[1][1]):
        #         if y_ <= right_range_[0][0]:
        #             right_range_[0] = [y_, x_]
        #         elif y_ >= right_range_[1][0]:
        #             right_range_[1] = [y_, x_]

        hash_connected_set_ = set(tuple(pos_) for pos_ in connected_set_)

        for x_ in range(bb_[0][1], bb_[1][1], 1):
            if self.__judge_boundary([bb_[0][0], x_], Direction.Down, hash_connected_set_):
                top_range_[0] = [bb_[0][0], x_]
                break
        for x_ in range(bb_[1][1], bb_[0][1], -1):
            if self.__judge_boundary([bb_[0][0], x_], Direction.Down, hash_connected_set_):
                top_range_[1] = [bb_[0][0], x_]
                break
        for x_ in range(bb_[0][1], bb_[1][1], 1):
            if self.__judge_boundary([bb_[1][0], x_], Direction.Up, hash_connected_set_):
                bot_range_[0] = [bb_[1][0], x_]
                break
        for x_ in range(bb_[1][1], bb_[0][1], -1):
            if self.__judge_boundary([bb_[1][0], x_], Direction.Up, hash_connected_set_):
                bot_range_[1] = [bb_[1][0], x_]
                break
        for y_ in range(bb_[0][0], bb_[1][0], 1):
            if self.__judge_boundary([y_, bb_[0][1]], Direction.Right, hash_connected_set_):
                left_range_[0] = [y_, bb_[0][1]]
                break
        for y_ in range(bb_[1][0], bb_[0][0], -1):
            if self.__judge_boundary([y_, bb_[0][1]], Direction.Right, hash_connected_set_):
                left_range_[1] = [y_, bb_[0][1]]
                break
        for y_ in range(bb_[0][0], bb_[1][0], 1):
            if self.__judge_boundary([y_, bb_[1][1]], Direction.Left, hash_connected_set_):
                right_range_[0] = [y_, bb_[1][1]]
                break
        for y_ in range(bb_[1][0], bb_[0][0], -1):
            if self.__judge_boundary([y_, bb_[1][1]], Direction.Left, hash_connected_set_):
                right_range_[1] = [y_, bb_[1][1]]
                break

        if is_draw_:
            st_ = self.__shrink_times
            self.__draw_solid_box(top_range_[0], 10, (255, 0, 0), enlarge_=st_)
            self.__draw_solid_box(top_range_[1], 10, (255, 0, 0), enlarge_=st_)
            self.__draw_solid_box(bot_range_[0], 10, (255, 0, 0), enlarge_=st_)
            self.__draw_solid_box(bot_range_[1], 10, (255, 0, 0), enlarge_=st_)
            self.__draw_line(top_range_[0], top_range_[1], 3, (255, 0, 0), enlarge_=st_)
            self.__draw_line(top_range_[1], bot_range_[1], 3, (255, 0, 0), enlarge_=st_)
            self.__draw_line(bot_range_[1], bot_range_[0], 3, (255, 0, 0), enlarge_=st_)
            self.__draw_line(bot_range_[0], top_range_[0], 3, (255, 0, 0), enlarge_=st_)

            self.__draw_solid_box(left_range_[0], 10, (0, 0, 255), enlarge_=st_)
            self.__draw_solid_box(left_range_[1], 10, (0, 0, 255), enlarge_=st_)
            self.__draw_solid_box(right_range_[0], 10, (0, 0, 255), enlarge_=st_)
            self.__draw_solid_box(right_range_[1], 10, (0, 0, 255), enlarge_=st_)
            self.__draw_line(left_range_[0], left_range_[1], 3, (0, 0, 255), enlarge_=st_)
            self.__draw_line(left_range_[1], right_range_[1], 3, (0, 0, 255), enlarge_=st_)
            self.__draw_line(right_range_[1], right_range_[0], 3, (0, 0, 255), enlarge_=st_)
            self.__draw_line(right_range_[0], left_range_[0], 3, (0, 0, 255), enlarge_=st_)

        fault_tolerance_ = 10
        if abs(top_range_[0][1] - bot_range_[0][1]) <= fault_tolerance_ and \
                abs(top_range_[1][1] - bot_range_[1][1]) <= fault_tolerance_:
            if abs(left_range_[0][0] - right_range_[0][0]) <= fault_tolerance_ and \
                    abs(left_range_[1][0] - right_range_[1][0]) <= fault_tolerance_:
                if manhattan_dis(top_range_[0], left_range_[0]) <= fault_tolerance_ and \
                        manhattan_dis(top_range_[1], right_range_[0]) <= fault_tolerance_ and \
                        manhattan_dis(bot_range_[0], left_range_[1]) <= fault_tolerance_ and \
                        manhattan_dis(bot_range_[1], right_range_[1]) <= fault_tolerance_:
                    return True
        return False

    def __locate_inner_line(self, range_, dir_):  # 定位某方向（根据 range_ 和 dir_）的内部线位置
        state = 0
        for pos in range_:
            if state == 0 and self.__judge_line(pos, dir_):  # 进入外部线
                state = 1
            elif state == 1 and not self.__judge_line(pos, dir_):  # 出外部线
                state = 2
            elif state == 2 and self.__judge_line(pos, dir_):  # 进入内部线
                state = 3
            elif state == 3 and not self.__judge_line(pos, dir_):  # 出内部线
                return pos

    def get_inner_image(self):
        return self.__inner_img

    def get_drew_image(self):
        return self.__raw_color_img

    def get_erode_image(self):
        return self.__find_inner_table_base_img

    def get_shrink_image(self):
        return self.__shrink_inner_img

    def get_shrink_bin_image(self):
        return self.__shrink_bin_inner_img

    def extract_edge_tables(self):
        img_ = self.__inner_img
        br_tables_, tr_tables_, bl_tables_, tl_tables_ = [], [], [], []
        self.__brush_color = (255, 0, 0)
        br_tables_ = self.locate_table_at_the_bottom_right(img_)
        # print('br_tables: ', br_tables_, get_table_type(br_tables_))
        if get_table_type(br_tables_) != TableType.Full:
            self.__brush_color = (0, 255, 0)
            tr_tables_ = self.locate_table_at_the_bottom_right(cv2.flip(img_, 0), 0)
            # print('tr_tables_: ', tr_tables_, get_table_type(tr_tables_))
            if get_table_type(br_tables_) != TableType.FullWidth:
                self.__brush_color = (0, 0, 255)
                bl_tables_ = self.locate_table_at_the_bottom_right(cv2.flip(img_, 1), 1)
                # print('bl_tables_: ', bl_tables_, get_table_type(bl_tables_))
            if get_table_type(tr_tables_) != TableType.FullWidth:
                self.__brush_color = (255, 0, 255)
                tl_tables_ = self.locate_table_at_the_bottom_right(cv2.flip(img_, -1), -1)
                # print('tl_tables_: ', tl_tables_, get_table_type(tl_tables_))

        all_tables_ = br_tables_
        for box_ in tr_tables_:
            all_tables_.append(correct_flipped_box(box_, img_.shape[:2], 0))
        for box_ in bl_tables_:
            all_tables_.append(correct_flipped_box(box_, img_.shape[:2], 1))
        for box_ in tl_tables_:
            all_tables_.append(correct_flipped_box(box_, img_.shape[:2], -1))

        return all_tables_

    def extract_inner_tables(self, visited_):
        img_ = self.__find_inner_table_base_img
        h_, w_ = img_.shape[:2]
        all_tables_ = []
        for y_ in range(0, h_, 1):
            for x_ in range(0, w_, 1):
                if is_white(img_, [y_, x_]) or visited_[y_, x_]:
                    visited_[y_, x_] = True
                else:
                    cur_connected_set_ = []
                    # self.__dfs(img_, visited_, x_, y_, cur_connected_set_)
                    self.__dfs_on_stack(img_, visited_, y_, x_, cur_connected_set_)
                    if len(cur_connected_set_) != 0:
                        bb_ = calc_bounding_box(cur_connected_set_)
                        if self.__judge_inner_table(cur_connected_set_, True):
                            all_tables_.append(bb_)
                            # print('inner')
                        visited_[bb_[0][0]:bb_[1][0], bb_[0][1]:bb_[1][1]] = True
                        # self.__draw_line(bb_[0], bb_[1], 5, (0, 255, 0))

        return all_tables_

    def extract_tables(self):
        all_edge_tables_ = self.extract_edge_tables()

        visited_ = np.zeros(self.__shrink_inner_img.shape[:2], dtype=bool)
        st_ = self.__shrink_times
        # 将已检测到的边界上的表格位置标记为“已访问”
        for box_ in all_edge_tables_:
            # for y_ in range(box_[0][0], box_[1][0], 1):
            #     for x_ in range(box_[0][1], box_[1][1], 1):
            #         visited_[y_, x_] = True
            visited_[box_[0][0]//st_:box_[1][0]//st_, box_[0][1]//st_:box_[1][1]//st_] = True

        all_inner_tables_ = self.extract_inner_tables(visited_)
        all_inner_tables_ = [[[axis * st_ for axis in pos] for pos in box] for box in all_inner_tables_]
        all_tables_ = all_edge_tables_ + all_inner_tables_
        for i_ in range(len(all_tables_)):
            all_tables_[i_][0][0] += self.__inner_offset[0]
            all_tables_[i_][1][0] += self.__inner_offset[0]
            all_tables_[i_][0][1] += self.__inner_offset[1]
            all_tables_[i_][1][1] += self.__inner_offset[1]

        all_table_sub_images_ = []
        for box_ in all_tables_:
            all_table_sub_images_.append(self.__raw_img[box_[0][0]:box_[1][0], box_[0][1]:box_[1][1]])
        return all_table_sub_images_

    def probe_ray(self, bin_img_, pos_, dir_, ft_width_=0, min_len_=20, step_=1, flip_=2,
                  is_draw_=False):  # 探查当前位置是否存在某一方向的射线
        if min_len_ < 0:
            min_len_ = 1e7

        if dir_ == Direction.Up:
            if pos_[0] == 0:
                return False, pos_
            range_ = range(pos_[0], max(pos_[0] - min_len_, -1), -step_)
        elif dir_ == Direction.Down:
            if pos_[0] == bin_img_.shape[0] - 1:
                return False, pos_
            range_ = range(pos_[0], min(pos_[0] + min_len_, bin_img_.shape[0]), step_)
        elif dir_ == Direction.Left:
            if pos_[1] == 0:
                return False, pos_
            range_ = range(pos_[1], max(pos_[1] - min_len_, -1), -step_)
        elif dir_ == Direction.Right:
            if pos_[1] == bin_img_.shape[1] - 1:
                return False, pos_
            range_ = range(pos_[1], min(pos_[1] + min_len_, bin_img_.shape[1]), step_)
        else:
            raise Exception('probe_ray::dir_ param error', dir_)

        trace_set_ = []
        y_, x_ = pos_[0], pos_[1]
        if bool(dir_ & Direction.Vertical):
            for y_ in range_:
                flag_ = False
                for ft_x in range(max(0, x_ - ft_width_), min(bin_img_.shape[1], x_ + ft_width_ + 1), 1):
                    flag_ = flag_ | is_black(bin_img_, [y_, ft_x])
                    trace_set_.append(correct_flipped_pos([y_, ft_x], self.__inner_img.shape[:2], flip_))
                if not flag_:
                    if is_draw_:
                        for pos_ in trace_set_:
                            self.__draw_solid_box(pos_, 1, (0, 0, 255))
                    return False, [y_, x_]
        else:
            for x_ in range_:
                flag_ = False
                for ft_y in range(max(0, y_ - ft_width_), min(bin_img_.shape[0], y_ + ft_width_ + 1), 1):
                    flag_ = flag_ | is_black(bin_img_, [ft_y, x_])
                    trace_set_.append(correct_flipped_pos([ft_y, x_], self.__inner_img.shape[:2], flip_))
                if not flag_:
                    if is_draw_:
                        for pos_ in trace_set_:
                            self.__draw_solid_box(pos_, 1, (0, 0, 255))
                    return False, [y_, x_]
        if is_draw_:
            for pos_ in trace_set_:
                self.__draw_solid_box(pos_, 1, (0, 255, 0))
        return True, [y_, x_]

    def find_ray_end(self, bin_img_, pos_, dir_):
        flag_, pos_ = self.probe_ray(bin_img_, pos_, dir_, 2, 10)
        while flag_:
            flag_, pos_ = self.probe_ray(bin_img_, pos_, dir_, 2, 10)
        return pos_

    # 定位图片右下角的表格
    def locate_table_at_the_bottom_right(self, img_, flip_=2):
        h_, w_ = img_.shape[:2]
        y_, x_ = h_ - 1, w_ - 1

        # 向左寻找入口
        while x_ >= 0 and is_white(img_, [y_, x_]):
            x_ -= 1
        if x_ < 0:  # 未找到入口
            return []
        enter_x_ = x_

        # 优先向上走，无上方的路便向左走，确定上边界位置
        while x_ >= 1 and y_ >= 1 and (is_black(img_, [y_ - 1, x_]) or is_black(img_, [y_, x_ - 1])):
            if is_black(img_, [y_ - 1, x_]):
                y_ -= 1
            else:
                x_ -= 1
        table_t_, table_l_ = y_, x_
        table_l_4_L_shaped_ = self.find_ray_end(img_, [table_t_, table_l_], Direction.Left)[1]  # 用于 “_|” 型堆叠表格的上表格的左边界

        '''
        # 优先向左走，确定左边界位置
        y_, x_ = h_ - 1, enter_x_
        while x_ >= 1 and y_ >= 1 and (is_black(img_, [y_ - 1, x_]) or is_black(img_, [y_, x_ - 1])):
            if is_black(img_, [y_, x_ - 1]):
                x_ -= 1
            else:
                y_ -= 1    
        '''

        # 从刚才找到的点出发，优先向左走，无左方的路便向下走，确定左边界位置
        can_left = self.probe_ray(img_, [y_, x_], Direction.Left, 3, 1)[0]
        can_down = self.probe_ray(img_, [y_, x_], Direction.Down, 3, 1)[0]
        while x_ >= 1 and y_ <= h_ - 2 and (can_left or can_down):
            if can_left:
                x_ -= 1
            else:
                y_ += 1
            can_left = self.probe_ray(img_, [y_, x_], Direction.Left, 3, 1)[0]
            can_down = self.probe_ray(img_, [y_, x_], Direction.Down, 3, 1)[0]
        table_t_, table_l_ = min(table_t_, y_), min(table_l_, x_)
        self.__draw_solid_box(correct_flipped_pos([table_t_, table_l_], self.__inner_img.shape[:2], flip_), radius_=10)

        if h_ - table_t_ < self.__min_table_size[0] or w_ - table_l_ < self.__min_table_size[1]:  # 区域过小，判断为非表格
            return []

        fault_tolerance_ = 10  # 容错像素量，用于规避 表格边界锯齿/边界与图片边缘断连 造成的错误
        if table_t_ < fault_tolerance_ and table_l_ < fault_tolerance_:  # 整张图均为表格
            return self.split_vertically_adjacent_equal_width_tables(img_, [table_t_, table_l_], [h_, w_])

        # is_exist_down_short_ray, _ = self.probe_ray(img_, [table_t_, table_l_], Direction.Down, 5, 100, 1, flip_,
        #                                             is_draw_=True)
        # is_exist_right_short_ray, _ = self.probe_ray(img_, [table_t_, table_l_], Direction.Right, 5, 100, 1, flip_,
        #                                              is_draw_=True)
        is_exist_down_ray = \
        self.probe_ray(img_, [table_t_ + fault_tolerance_, table_l_], Direction.Down, 5, -1, 10, flip_,
                       is_draw_=True)[0]
        is_exist_right_ray = \
        self.probe_ray(img_, [table_t_, table_l_ + fault_tolerance_], Direction.Right, 5, -1, 10, flip_,
                       is_draw_=True)[0]
        if is_exist_right_ray:  # 单表格
            return [[[table_t_, table_l_], [h_, w_]]]

        # 检测根据“各分量最小点”的位置，从图片边缘向内是否存在短射线（即表格的边界），若短射线紧靠图片边缘则默认存在该射线
        is_exist_left_ray_from_edge = self.probe_ray(img_, [table_t_, (w_ - 1) - fault_tolerance_], Direction.Left, 5,
                                                     self.__min_table_size[1], flip_=flip_, is_draw_=True)[0] if table_t_ > fault_tolerance_ else True
        is_exist_up_ray_from_edge = self.probe_ray(img_, [(h_ - 1) - fault_tolerance_, table_l_], Direction.Up, 5,
                                                   self.__min_table_size[0], flip_=flip_, is_draw_=True)[0] if table_l_ > fault_tolerance_ else True

        if is_exist_left_ray_from_edge and is_exist_up_ray_from_edge:  # 相邻两表格呈 “_|” 型
            return self.split_vertically_adjacent_l_shaped_tables(img_, [table_t_, table_l_], [h_, w_],
                                                                  [table_t_, table_l_4_L_shaped_])
        return []

    # 垂直分割两个等宽的表格
    def split_vertically_adjacent_equal_width_tables(self, bin_img_, pos_tl_, pos_br_):
        enter_x_ = -1
        # 自左上角，再向右偏移 10 像素（为了跳过左边界竖线）向右出发，找到首个竖线
        for x_ in range(pos_tl_[1] + 10, pos_br_[1], 1):
            if self.probe_ray(bin_img_, [pos_tl_[0], x_], Direction.Down)[0]:
                enter_x_ = x_
                break
        if enter_x_ < 0:
            print('split_vertically_adjacent_equal_width_tables::enter_x_ error')
            return pos_br_[0]

        y_ = pos_tl_[0]  # 防警告
        # 从上述找到的首个竖线位置，从上往下直到该竖线不再延伸
        for y_ in range(pos_tl_[0] + 3, pos_br_[0], 3):
            if not self.probe_ray(bin_img_, [y_, enter_x_], Direction.Down, 2, 3)[0]:
                break

        if abs(y_ - pos_br_[0]) < 10 or abs(y_ - pos_tl_[0]) < 10:  # 只有一个表格
            return [[pos_tl_, pos_br_]]

        # 有两个表格
        top_pos_br_ = [y_, pos_br_[1]]
        bot_pos_tl_ = [y_, pos_tl_[1]]
        return [[pos_tl_, top_pos_br_], [bot_pos_tl_, pos_br_]]

    # 分割“_|"状的两个表格
    def split_vertically_adjacent_l_shaped_tables(self, bin_img_, pos_tl_, pos_br_, top_pos_tl_):
        y_ = top_pos_tl_[0]  # 防警告
        # 从上方表格的左边界向下遍历，每次都向左找是否存在射线，若存在则到了两表格的分界线
        for y_ in range(top_pos_tl_[0], pos_br_[0], 1):
            if self.probe_ray(bin_img_, [y_, top_pos_tl_[1]], Direction.Left)[0]:
                break
        if abs(y_ - pos_br_[0]) < 10:
            print('split_vertically_adjacent_l_shaped_tables::only one table')
            return [[pos_tl_, pos_br_]]

        top_pos_br_ = [y_, pos_br_[1]]
        bot_pos_tl_ = [y_, pos_tl_[1]]
        return [[top_pos_tl_, top_pos_br_], [bot_pos_tl_, pos_br_]]
