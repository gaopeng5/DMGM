#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_camera_intrinsic():

    VIEW_WIDTH = 1920       # int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = 1080      # int(sensor.attributes['image_size_y'])
    VIEW_FOV = 120          # int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        
    return matrix_k

def draw_matching(data, y, mm, co_index, save_path, method):

    img_s = Image.open('{}.png'.format(data.name_s[0]))
    img_t = Image.open('{}.png'.format(data.name_t[0]))
    frame = os.path.split(data.name_s[0])[1]
    # stack two images vertically
    if data.switched[0]:
        img = Image.fromarray(np.concatenate((np.array(img_t), 
                                              np.array(img_s))))
    else:
        img = Image.fromarray(np.concatenate((np.array(img_s),
                                              np.array(img_t))))

    fig=plt.figure(num=None, figsize=(16, 12), dpi=120, 
                   facecolor='w', edgecolor='k')
    plt.imshow(img)
    
    # get camera intrinsic matrix K
    K = get_camera_intrinsic()
    # data.pos_s contains 3D positions of bounding boxes in camera coordinates
    # pos2d contains 2D pixel position of bounding boxes in figures 
    pos_sen_s = np.dot(K, np.transpose(data.pos_s.cpu().numpy()))
    x_s = pos_sen_s[0, :] / pos_sen_s[2, :]
    y_s = pos_sen_s[1, :] / pos_sen_s[2, :]

    pos_sen_t = np.dot(K, np.transpose(data.pos_t.cpu().numpy()))
    x_t = pos_sen_t[0, :] / pos_sen_t[2, :]         
    y_t = pos_sen_t[1, :] / pos_sen_t[2, :]
    
    # 1920 is picture width, 1080 is picture height 
    if data.switched[0]:
        tmp_x = x_s
        tmp_y = y_s
        x_s = x_t
        y_s = y_t
        x_t = tmp_x
        y_t = tmp_y + 1080
    else:
        y_t = y_t + 1080
    # draw dot for all detected objects
    plt.plot(x_s, y_s, 'ro', markersize=4)
    plt.plot(x_t, y_t,'bx', markersize=4)
    
    gt = y[1]
    pred = mm
    temp = []
    co_index=y[0] * co_index
    # co_index contains which correspondence we keep
    if data.switched[0]:
        for i, data in enumerate(co_index):
            temp.append(i)
            if pred[data]==gt[data]:
                index_s=int(data.cpu().numpy())
                index_t=int(gt[data].cpu().numpy())
                # draw green line for correct correspondence
                plt.plot([x_s[index_t], x_t[index_s]], 
                         [y_s[index_t], y_t[index_s]],'g', linewidth=2)
                
            if (pred[data]!=gt[data]) and (gt[data]!=100) :
                index_s=int(data.cpu().numpy())
                index_t=int(pred[data].cpu().numpy())
                # draw red line for correct correspondence
                plt.plot([x_s[index_t], x_t[index_s]],
                         [y_s[index_t], y_t[index_s]], 'r', linewidth=2)
    else:
        for i, data in enumerate(co_index):
            temp.append(i)
            if pred[data]==gt[data]:
                index_s=int(data.cpu().numpy())
                index_t=int(gt[data].cpu().numpy())
                # draw green line for correct correspondence
                plt.plot([x_s[index_s], x_t[index_t]],
                         [y_s[index_s], y_t[index_t]], 'g', linewidth=2)
                
            if (pred[data]!=gt[data]) and (gt[data]!=100) :
                index_s=int(data.cpu().numpy())
                index_t=int(pred[data].cpu().numpy())
                # draw red line for correct correspondence
                plt.plot([x_s[index_s], x_t[index_t]],
                         [y_s[index_s], y_t[index_t]], 'r', linewidth=2)
    
    save_fig_path = os.path.join(save_path, method, 'demo')
    if os.path.exists(save_fig_path) == False:
        os.mkdir(save_fig_path)
    # save the figure to file    
    fig.savefig(os.path.join(save_fig_path, '{}.png'.format(frame)))   
    plt.close(fig)  
    img_s.close()
    img_t.close()
