# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:38:55 2021

@author: MaxGr
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import re

import pickle
import collections

import scipy
from scipy import signal, spatial

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

import pandas as pd
import glob

import time



trajectory_path = "./trajectory/"
trajectory_list = os.listdir(trajectory_path)

class_path = "./class/"
class_list = os.listdir(class_path)



scene = 1


# traffic_trajectory_alig = np.load("Traffic_trajectory_aligned.npy",allow_pickle=True)
    
trajectory_list_i = glob.glob('./trajectory/Test'+str(scene)+'_*')
class_list_i = glob.glob('./class/Test'+str(scene)+'_*')






'''
Road size: 100m x 26m,
scale to 1000x 260 virtual map
Mapping raw video view to satellite view in scale(10)
map 4 points of a to 4 points of b
Triangle(3) or quadrilateral(4)
'''

# Test 1 i10_7th St
H,W = 1000, 260

raw_video = cv2.imread('Test1_part1_Moment.jpg')
plt.imshow(raw_video)

GPSMap_raw = cv2.imread('scene_1.png')
plt.imshow(GPSMap_raw)

rec_map = np.zeros((H, W))
plt.imshow(rec_map)



# Key points from video
# Test 1
keypoints_location = [(260, 670),
                      (650, 190), 
                      (920, 190),
                      (1220, 670)]

# GPS view
# Test 1
keypoints_location_GPS = [(502, 1507),
                          (540, 1044), 
                          (660, 1055),
                          (625, 1520)]



# # Test 1 i10_16th St
# h,w = 1400, 460

# raw_video = cv2.imread('Test5_part_1_Moment.jpg')
# plt.imshow(raw_video)

# GPSMap_raw = cv2.imread('GPSMap_Test5_16thSt.png')
# plt.imshow(GPSMap_raw)

# GPSMap = np.zeros((h, w))
# plt.imshow(GPSMap)



# # Test 5
# keypoints_location = [(0, 735),
#                       (1252, 308), 
#                       (1687, 320),
#                       (1267, 920)]

# # GPS view
# # Test 5
# keypoints_location_GPS = [(322, 787),
#                           (331, 21), 
#                           (555, 23),
#                           (543, 787)]



# Project to satellite view
keypoints_location_ref = [(0, H),
                          (0, 0), 
                          (W, 0),
                          (W, H)]



keypoints_location = [[*row] for row in keypoints_location]
keypoints_location_ref = [[*row] for row in keypoints_location_ref]
keypoints_location_GPS = [[*row] for row in keypoints_location_GPS]

keypoints_location = np.array(keypoints_location).astype(np.float32)
keypoints_location_ref = np.array(keypoints_location_ref).astype(np.float32)
keypoints_location_GPS = np.array(keypoints_location_GPS).astype(np.float32)




















def GPS_Alignment(image, keypoints_location, keypoints_location_ref, MAP_SIZE_REF):
    '''
    According to perspective transformation, 
    generate matrix M to mapping keypoints_location to keypoints_location_ref

    '''
    #MAP_SIZE_REF = (1000,1000)
    M = cv2.getPerspectiveTransform(keypoints_location, keypoints_location_ref)
    dendrite_aligned = cv2.warpPerspective(image, M, MAP_SIZE_REF)
    
    return M, dendrite_aligned

M, video_aligned = GPS_Alignment(raw_video, keypoints_location, keypoints_location_ref, MAP_SIZE_REF=(W,H))
plt.imshow(video_aligned)
M2, GPSMap_aligned = GPS_Alignment(GPSMap_raw, keypoints_location_GPS, keypoints_location_ref, MAP_SIZE_REF=(W,H))
plt.imshow(GPSMap_aligned)

cv2.imwrite('GPSMap.tif', GPSMap_aligned)





trajectory_list_i[1]
class_list_i[1]






def txt_to_dataframe(file_path):
    '''
    Read raw data from files and return a dictionary: 
        {frame_id: 
            {object_id: 
                [frame_id, object_id, bbox_top, bbox_left, bbox_w, bbox_h, -1]
            }
        }
    '''
    with open(file_path, 'r') as reader:
        print(file_path)
        content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(np.float32)
    
    return content
    
traffic_trajectory = txt_to_dataframe(trajectory_list_i[1])    
traffic_class      = txt_to_dataframe(class_list_i[1])
    
traffic_class = np.insert(traffic_class, 1, np.zeros((len(traffic_class))), axis=1)    
    




def file_merge(traffic_trajectory, traffic_class):
    
    for i in range(len(traffic_trajectory)):
        check_list = np.where(traffic_class[:,0]==traffic_trajectory[i,0])[0]
        x1,y1 = (traffic_trajectory[i,2]+traffic_trajectory[i,4]//2), (traffic_trajectory[i,3]+traffic_trajectory[i,5]//2)
        x2,y2 = (traffic_class[check_list,1]+traffic_class[check_list,3]//2), (traffic_class[check_list,2]+traffic_class[check_list,4]//2)
        
        if len(np.where( np.abs(x2-x1)+np.abs(y2-y1) < 15 )[0]) > 0:
            class_i = np.where( np.abs(x2-x1)+np.abs(y2-y1) < 15 )[0]
        
            traffic_trajectory[i,6] = traffic_class[check_list[class_i[0]], 5]
            
        if i%100 == 0: print(i)
        
    return traffic_trajectory

# a = file_merge(traffic_trajectory, traffic_class)
    
    
    
    
    
    
    
    
    
    
    

def trajectory_alignment(dataframe, M, traffic_class=[]):
    '''
    Project trajectory to real world according to perspective matrix M
    
    Output format:
        
        0: Time information
        1: Frame index
        2: Target index
        3: Target x location in top view
        4: Target y location in top view
        5: Target z location, optional parameter, used for reverse transformation
        6: Frame quantity
        7: Overall mean velocity 
        8: Instantaneous Velocity
        9: Trajectory prediction score
        10: 
    
    '''
    
    # Get center point
    bbox_location = np.zeros((dataframe.shape))
    bbox_location[:,0] = (dataframe[:,2]+dataframe[:,4]//2)
    bbox_location[:,1] = (dataframe[:,3]+dataframe[:,5]//2)
    bbox_location[:,2] = 1
    bbox_location = bbox_location[:,0:3]


    bbox_location_ref = np.zeros((dataframe.shape))
    
    for i in range(bbox_location_ref.shape[0]):
        # Tansform
        bbox_location_ref[i,2] = M[0,0]*bbox_location[i,0] + M[0,1]*bbox_location[i,1] + M[0,2]*bbox_location[i,2]
        bbox_location_ref[i,3] = M[1,0]*bbox_location[i,0] + M[1,1]*bbox_location[i,1] + M[1,2]*bbox_location[i,2]
        bbox_location_ref[i,4] = M[2,0]*bbox_location[i,0] + M[2,1]*bbox_location[i,1] + M[2,2]*bbox_location[i,2]
        
        bbox_location_ref[i,2] = bbox_location_ref[i,2] / bbox_location_ref[i,4]
        bbox_location_ref[i,3] = bbox_location_ref[i,3] / bbox_location_ref[i,4]
        

        if i%10000 == 0: print(i)

    bbox_location_ref[:,0:2] = dataframe[:,0:2]
    bbox_location_ref[:,4]   = dataframe[:,6]

    return bbox_location_ref
    

traffic_trajectory_alig = trajectory_alignment(traffic_trajectory, M, traffic_class)
traffic_class_alig = trajectory_alignment(traffic_class, M, traffic_class)







'''
Set index

'''
index_Time = 0
index_Frame = 1
index_Target = 2
index_x = 3 
index_y = 4
index_class = 5
index_FrameQuantity = 6
index_Mean_V = 7
index_Instantaneous_V = 8
index_Traj_pred_score = 9



def formating(dataframe): 
    # Increase 3 column on Right, 
    # Increase 1 column on Left
    dataframe = np.hstack((dataframe, np.zeros((len(dataframe),3))))
    dataframe = np.hstack((np.zeros((len(dataframe),1)),dataframe))
    
    return dataframe

traffic_trajectory_alig = formating(traffic_trajectory_alig)
traffic_class_alig =      formating(traffic_class_alig)



def delet_out_range(dataframe, h, w):
    '''
    Delet out ranged target,
    Focus on ROI target
    
    '''
    x = dataframe[:,index_x]
    y = dataframe[:,index_y]
    
    out_range = np.unique(np.hstack((np.where(x<0)[0], np.where(x>w)[0], np.where(y<0)[0], np.where(y>h)[0])))
    
    dataframe = np.delete(dataframe, out_range, axis=0)
    
    return dataframe

traffic_trajectory_alig = delet_out_range(traffic_trajectory_alig,H,W)
traffic_class_alig      = delet_out_range(traffic_class_alig,H,W)







# point out all aligned trajectory to aligned view
traffic_trajectory_alig_points = [tuple(l) for l in traffic_trajectory_alig[:,3:5].astype(int)]
# traffic_trajectory_alig_points = [tuple(l) for l in target_i[:,2:4].astype(int)]
for point in traffic_trajectory_alig_points:
 	cv2.circle(GPSMap_aligned, point, 1, (0, 255, 0), -1)

plt.imshow(GPSMap_aligned)
cv2.imwrite('GPSMap_mask.tif', GPSMap_aligned)

    

def get_target_i(dataframe, target_index):
    '''
    Get target i from all frames
    
    '''

    target_i_index = np.where(dataframe[:,index_Target]==target_index)[0]
    target_i       = dataframe[ target_i_index ]
    
    return target_i, target_i_index


def get_frame_i(dataframe, frame_index):
    '''
    Get frame i from all frames
    
    '''

    frame_i_index = np.where(dataframe[:,index_Frame]==frame_index)[0]
    frame_i       = dataframe[ frame_i_index ]
    
    return frame_i, frame_i_index

# #
# target_i = get_target_i(traffic_trajectory_alig, 150)[0]

# traffic_trajectory_alig_points = [tuple(l) for l in target_i[:,2:4].astype(int)]
# for point in traffic_trajectory_alig_points:
#  	cv2.circle(GPSMap_aligned, point, 5, (0, 0, 255), -1)

# plt.imshow(GPSMap_aligned)
# cv2.imwrite('GPSMap_trajectory.png', GPSMap_aligned)


# #
# t1 = get_target_i(traffic_trajectory_alig, 390)[0]
# t2 = get_target_i(traffic_trajectory_alig, 396)[0]
# t3 = get_target_i(traffic_trajectory_alig, 1374)[0]
# t4 = get_target_i(traffic_trajectory_alig, 3295)[0]
# t5 = get_target_i(traffic_trajectory_alig, 2587)[0]

# t_all = np.vstack((t1,t2,t3,t4,t5))

# plt.scatter(t5[:,3], t5[:,4])
# plt.plot(t1[:,3], t1[:,4])
# plt.xlim(0, 260)
    



def trajectory_analysis(target_i):
    x,y = target_i[:,index_x], target_i[:,index_y]
    x_ln, y_ln = x.reshape(-1,1), y.reshape(-1,1)
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_ln,y_ln)
    pred_score = model.score(x_ln,y_ln)
    model.predict(x_ln)
    
    a=model.intercept_
    b=model.coef_
    y_pred=model.predict(x_ln)
    
    # calculate mean  
    x_mean = np.mean(x)
    # calculate variance   
    x_std = np.std(x)
    # standardize X  
    x_norm = (x-x_mean)/x_std + x_mean
        
    return [x_norm, y_pred, pred_score]





# abnormal_target = traffic_trajectory_alig[traffic_trajectory_alig[:,8] < 0.8]

# t1 = get_target_i(traffic_trajectory_alig, 85557)[0]
# # plt.scatter(t1[:,2], t1[:,3])
# # plt.plot(t1[:,2], t1[:,3])
# # plt.xlim(0, 260)
   


# target_i = t1
# x,y = target_i[:,2], target_i[:,3]
# x_norm,y_pred,pred_score = trajectory_analysis(target_i)


# plt.plot(x_norm, y, 'ko')
# # plt.scatter(x, y)
# # plt.plot(x, y_pred)



# x_smooth = scipy.signal.savgol_filter(x_norm, window_length=21, polyorder=4)
# # y_smooth = smooth(target_i, 5)[:,3]

# plt.plot(x_smooth, y, linewidth=2.0, color='r')
# plt.xlim(min(x_norm)-5, max(x_norm)+5)





    
def smooth_xy(target_i, window_size):
    stride = window_size//2 
    for v in range(stride, len(target_i)-stride):

        X = target_i[v+stride, index_x] - target_i[v-stride, index_x]
        Y = target_i[v+stride, index_y] - target_i[v-stride, index_y]

        Vmean_mph = np.sqrt(X**2 + Y**2)/((window_size-1)/15) *3.6    /16
        
        target_i[v,index_Instantaneous_V] = Vmean_mph
        target_i[0:stride+1,index_Instantaneous_V] = target_i[stride,index_Instantaneous_V]
        target_i[(-stride-1):,index_Instantaneous_V] = target_i[-stride-1,index_Instantaneous_V]
            
    return target_i    


def smooth_v(target_i, window_size):
    stride = window_size//2 
    for v in range(stride, len(target_i)-stride):

        V = target_i[v-stride: v+stride+1, 7]
        
        V_norm = (V-np.mean(V))/np.std(V) + np.mean(V)
        
        # Vmean_mph = np.sqrt(V**2)/((window_size-1)/15) *3.6    /16
        
        target_i[v, 7] = V_norm[2]
        target_i[0:stride+1,7] = target_i[stride,7]
        target_i[(-stride-1):,7] = target_i[-stride-1,7]
            
    return target_i    
    

# dataframe = traffic_trajectory_alig

def get_info(dataframe):
    '''
    Get all information for necessary
    
    '''
    delet_list = []
    
    n = np.unique(dataframe[:, index_Target])
    for i in n:
        target_i_index = get_target_i(dataframe, i)[1]
        target_i       = get_target_i(dataframe, i)[0]
        length = len(target_i_index)
        
        if length == 0:
            continue
        
        
        
        
        # Delet odd target
        # 1. FPS missing
        # 2. FPS too short
        FPS = target_i[-1, index_Frame] - target_i[0, index_Frame]
        if (FPS < 60) or (np.round(FPS/length) > 2):
            delet_list.append(target_i_index)
            continue
        dataframe[target_i_index, index_FrameQuantity] = FPS

        
        # Get speed
        T = FPS/30
        X = target_i[-1, index_x]/10 - target_i[0, index_x]/10
        Y = target_i[-1, index_y]/10 - target_i[0, index_y]/10
        S = np.sqrt(X**2 + Y**2)
        Vmean_kmh = S/T *3.6
        Vmean_mph = Vmean_kmh/1.6
        dataframe[target_i_index, index_Mean_V] = Vmean_mph
        
        
        # Get trajectory pred score
        pred_score = trajectory_analysis(target_i)[2]
        dataframe[target_i_index, index_Traj_pred_score] = pred_score

        
        # Get Instantaneous Velocity for all target in real scale 
        target_smooth = smooth_xy(target_i, window_size=5)
        polyorder = 7
        window_length = length//2
        if window_length%2 == 0: window_length = window_length+1
        if window_length <= polyorder: polyorder = window_length-2
        V_inst = scipy.signal.savgol_filter(target_smooth[:,index_Instantaneous_V], window_length, polyorder)
        # target_i = smooth_v(target_i, 7)
        dataframe[target_i_index, index_Instantaneous_V] = V_inst
        
        
 
        # Get class
        check_list = np.where(traffic_class_alig[:,1]==target_i[1,1])[0]
        x1,y1 = target_i[1,3:5]
        x2,y2 = traffic_class_alig[ check_list ][:,3], traffic_class_alig[ check_list ][:,4]
        
        check = np.where( np.abs(x2-x1)+np.abs(y2-y1) < 20 )[0]
        if check.size > 0:
            target_class = traffic_class_alig[check_list[check], 5] 
            if len(target_class) > 1:
                target_class = target_class[0]

        dataframe[target_i_index, index_class] = target_class
        
        
        

        if i%100 == 0: print(str(i)+'|'+str(n[-1]))


    delet_list = [i for k in delet_list for i in k]
    dataframe = np.delete(dataframe, delet_list[::], axis=0)
        
        
    return dataframe

time_start = time.time()
traffic_trajectory_alig = get_info(traffic_trajectory_alig)
time_end = time.time()

print(time_end-time_start)
    
    
    
    
    
    

# Convert format
traffic_trajectory_alig = np.array(traffic_trajectory_alig, dtype=object)

# Adjust dtype
traffic_trajectory_alig[:,1:3] = np.int32((traffic_trajectory_alig[:,1:3]))
traffic_trajectory_alig[:,5:7] = np.int16((traffic_trajectory_alig[:,5:7]))


traffic_trajectory_alig[:,3:5] = np.float16((traffic_trajectory_alig[:,3:5]))
traffic_trajectory_alig[:,7:10] = np.float16((traffic_trajectory_alig[:,7:10]))


    
    
    
    
def get_time(dataframe):
    '''
    Get time information in real time coordinate
    
    '''
    
    init_time = datetime(2021, 3, 25, 11, 0, 13)
    print(init_time)
    
    for i in range(len(dataframe)):
        
        frame = dataframe[i, index_Frame]
        seconds = frame//30
        target_time = init_time + timedelta(seconds=seconds)

        dataframe[i, index_Time] = target_time
        
        if i%100 == 0: print(target_time, i)
        
    return dataframe

traffic_trajectory_alig = get_time(traffic_trajectory_alig)



    



# Save & load
save_path = './save/'

finalfile_npy = trajectory_file.replace('.txt','.npy')
finalfile_csv = trajectory_file.replace('.txt','.csv')

np.save(save_path+finalfile_npy, traffic_trajectory_alig)
np.savetxt(save_path+finalfile_csv, traffic_trajectory_alig, fmt='%s', delimiter=',')

b = np.load('./save/Test1_part_1.npy',allow_pickle=True)
    


    
    



    
    
    
    

# # i=29

# # a = smooth(target_i,5)[:,7]
        
# # a = scipy.signal.savgol_filter(target_i[:,7],window_length=int(len(target_i)/2),polyorder=9)

# # a = smooth()

# # a = preprocessing.scale(target_i[:,7])  # 调用sklearn包的方法

# # plt.plot(target_i[:,0], np.round(target_i[:,7]))
# # plt.ylim(0, 120)

# # plt.plot(target_i[:,0], a)
# # plt.plot(target_i[:,0], smooth(target_i[:,7],3))

    
    


# '''
# Plot overview of speed statistic

# '''

# plot information
speed = traffic_trajectory_alig[:,7].astype(np.int32)
speed_freq = collections.Counter(speed)


plt.scatter(range(len(np.unique(traffic_trajectory_alig[:,7]))), np.unique(traffic_trajectory_alig[:,7]))
   
plt.scatter(np.unique(traffic_trajectory_alig[:,7]), range(len(np.unique(traffic_trajectory_alig[:,7]))))
    
plt.scatter(list(speed_freq.keys()), list(speed_freq.values()))



# # plot information
# pred = np.round(traffic_trajectory_alig[:,8],1)
# pred_freq = collections.Counter(pred)

# plt.scatter(range(len(np.unique(traffic_trajectory_alig[:,8]))), np.unique(traffic_trajectory_alig[:,8]))
# plt.scatter(np.unique(traffic_trajectory_alig[:,8]), range(len(np.unique(traffic_trajectory_alig[:,8]))))
# plt.scatter(list(pred_freq.keys()), list(pred_freq.values()))


# # a = traffic_trajectory_alig[np.where(traffic_trajectory_alig==min(traffic_trajectory_alig[:,6]))[0]]

# # np.where(traffic_trajectory_alig==max(traffic_trajectory_alig[:,6]))

# # np.mean(traffic_trajectory_alig[:,5])

    
# i=15

# current_target = np.where(traffic_trajectory_alig[:,1]==i)[0]

# # current_target = np.where(traffic_trajectory_alig[:,5]==max(traffic_trajectory_alig[:,5]))[0]

# target_i = traffic_trajectory_alig[current_target]
    
# # b = traffic_trajectory_alig[np.int32(target_list)]

# plt.plot(target_i[:,0], np.round(target_i[:,7]))
# plt.ylim(0, 120)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# class Traffic_Trajectory_Aligned:
#     '''
#     Input the pre-processed dataframe (ex: traffic_trajectory_alig)
    
#     '''
    
#     def __init__(self, dataframe):
#         self.dataframe = dataframe
#         self.target_list = np.unique(self.dataframe[:,1])
#         self.length = len(self.target_list)

#     def get_target_i(self, index):
#         self.index = index
        
#         self.target_index = np.where(self.dataframe[:,1]==self.index)[0]
#         self.info = self.dataframe[ self.target_index ]
        
#         return self.info
        
        
#     def get_speed(self, index):
#         '''
#         Get speed of target i in real scale
        
#         '''
#         target_i = self.get_target_i(index)
#         if len(target_i) == 0:
#             return print('No target found')

#         return target_i[0,6]
    
    
#     def get_ins_speed(self, index):
#         '''
#         Get Instantaneous Velocity for target i in real scale
    
#         '''
#         target_i = self.get_target_i(index)
#         if len(target_i) == 0:
#             return print('No target found')
        
#         return target_i[:,7]
    

#     def IVVR(self):
#         print('Processing IVVR...')
#         count = 0
#         Vi = []
        
#         for i in self.target_list:
#             vi_max = max(self.get_ins_speed(i))
#             vi_min = min(self.get_ins_speed(i))
#             vi_mean = self.get_speed(i)
            
#             Vi.append(abs(vi_max-vi_min)/vi_mean)
            
#             if count%1000 == 0:
#                 print(str(count) + '/' + str(self.length))
            
#             count = count + 1
    
#         N = self.length
            
#         self.IVVR = np.sum(np.array(Vi))/N
        
#         print('IVVR = '+ str(self.IVVR))
        
#         return self.IVVR
    

#     def OVVR(self):
#         print('Processing OVVR...')
        
#         count = 0
#         Vi_mean = []
        
#         for i in self.target_list:
#             vi_mean = self.get_speed(i)
#             Vi_mean.append(vi_mean)
            
#             if count%1000 == 0:
#                 print(str(count) + '/' + str(self.length))
            
#             count = count + 1
        
#         N = self.length
#         Vi_mean = np.array(Vi_mean)
#         Vmean = np.sum(Vi_mean)/N
            
#         self.OVVR = np.sum(abs(Vi_mean-Vmean)/Vmean)/N
        
#         print('OVVR = '+ str(self.IVVR))
        
#         return self.OVVR
    
    
#     def OSR(self, speed_limit):
#         print('Processing OSR...')
        
#         count = 0
#         I = 0
        
#         for i in self.target_list:
#             vi_max = max(self.get_ins_speed(i))
#             if vi_max > speed_limit:
#                 I = I+1
            
#             if count%1000 == 0:
#                 print(str(count) + '/' + str(self.length))
            
#             count = count + 1
        
#         N = self.length

#         self.OSR = I/N
        
#         print('OSR('+str(speed_limit)+') = '+ str(self.OSR))
        
#         return self.OSR
        
    
    


# Traffic_All = Traffic_Trajectory_Aligned(traffic_trajectory_alig)

# Traffic_All.target_list

# Traffic_All.get_target_i(15)
# Traffic_All.get_speed(15)
# Traffic_All.get_ins_speed(15)

# Traffic_All.IVVR()
# Traffic_All.IVVR

# Traffic_All.OVVR()
# Traffic_All.OVVR

# Traffic_All.OSR(65)
# Traffic_All.OSR







# b = Traffic_All.info

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    