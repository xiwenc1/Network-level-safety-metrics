# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:30:13 2021

@author: MaxGr
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math


import folium
import pyproj
from pyproj import Proj, CRS, transform

from geopy.distance import geodesic
import geographiclib
from geographiclib.geodesic import Geodesic



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob 
from datetime import datetime,timedelta

import pickle
import collections

import scipy
from scipy import signal, spatial

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


import copy
# import sys
# sys.path.append('C://Users/MaxGr/Desktop/Python/PII/GPSMapping')


# # import metricExtract.py


# Project_PATH = os.path.dirname(os.path.abspath('__file__'))











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
index_TTC = 10




def scene(scene_id):
    '''
    Road size: 100m x 26m,
    scale to 1000x 260 virtual map
    Mapping raw video view to satellite view in scale(10)
    map 4 points of a to 4 points of b
    Triangle(3) or quadrilateral(4)
    '''
    if scene_id == 1:
        # Test 1 i10_7th St
        h,w = 1000, 260
        lane = 5
        scene_mask = cv2.imread('scene_1_mask.bmp')
        plt.imshow(scene_mask)
        
        
        raw_video = cv2.imread('Test1_part1_Moment.jpg')
        plt.imshow(raw_video)
        
        GPSMap_raw = cv2.imread('scene_1.png')
        plt.imshow(GPSMap_raw)
        
        rec_map = np.zeros((h, w))
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
        
        # Project to satellite view
        keypoints_location_ref = [(0, h),
                                  (0, 0), 
                                  (w, 0),
                                  (w, h)]
        
        # Time frame list
        time_list =  [datetime(2021, 11, 20,  10, 27, 45),
                      datetime(2021, 11, 20,  11, 27, 46),
                      datetime(2021, 11, 20,  12, 27, 46),
                      datetime(2021, 11, 20,  13, 27, 47),
                      datetime(2021, 11, 20,  14, 27, 48),
                      datetime(2021, 11, 20,  15, 27, 49),
                      datetime(2021, 11, 20,  16, 27, 49),
                      datetime(2021, 11, 20,  17, 27, 50)]
    
        
    if scene_id == 4:
        # Test 4 i10_5th Ave
        h,w = 800, 230
        lane = 5
        scene_mask = cv2.imread('scene_4_mask.bmp')
        plt.imshow(scene_mask)
        
        raw_video = cv2.imread('Test4_n_part2_Moment.jpg')
        plt.imshow(raw_video)
        
        GPSMap_raw = cv2.imread('scene_4.png')
        plt.imshow(GPSMap_raw)
        
        rec_map = np.zeros((h, w))
        # plt.imshow(GPSMap)

        # Key points from video
        # Test 1
        keypoints_location = [(385, 656),
                              (507, 193), 
                              (744, 207),
                              (1048, 595)]
        
        # GPS view
        # Test 1
        keypoints_location_GPS = [(527, 1253),
                                  (564, 517), 
                                  (777, 538),
                                  (734, 1253)]
        
        # Project to satellite view
        keypoints_location_ref = [(0, h),
                                  (0, 0), 
                                  (w, 0),
                                  (w, h)]
        
        # Time frame list
        time_list =  [datetime(2021, 12, 15, 7, 10, 12),
                      datetime(2021, 12, 15, 9, 10, 13),
                      datetime(2021, 12, 15, 11, 10, 14),
                      datetime(2021, 12, 15, 13, 10, 16),
                      datetime(2021, 12, 15, 15, 10, 17)]
        
        
        
        
    if scene_id == 5:
        # Test 1 i10_16th St
        h,w = 1000, 370
        lane = 7
        scene_mask = cv2.imread('scene_5_mask.bmp')
        plt.imshow(scene_mask)
        
        raw_video = cv2.imread('Test5_n_part1_Moment.jpg')
        plt.imshow(raw_video)
        
        GPSMap_raw = cv2.imread('scene_5.png')
        plt.imshow(GPSMap_raw)
        
        rec_map = np.zeros((h, w))
        # plt.imshow(GPSMap)
        
        # Test 5
        keypoints_location = [(160, 640),
                              (990, 345), 
                              (1402, 357),
                              (1153, 773)]
        
        # GPS view
        # Test 5
        keypoints_location_GPS = [(322, 732),
                                  (331, 195), 
                                  (534, 196),
                                  (537, 732)]
        
        # Project to satellite view
        keypoints_location_ref = [(0, h),
                                  (0, 0), 
                                  (w, 0),
                                  (w, h)]
        
        # Time frame list
        time_list =  [datetime(2021, 10, 25,  9, 39, 53),
                      datetime(2021, 10, 25, 11, 39, 54),
                      datetime(2021, 10, 25, 13, 39, 55),
                      datetime(2021, 10, 25, 15, 39, 55),
                      datetime(2021, 10, 25, 17, 39, 56)]
        
        
    if scene_id == 6:
        # Test 1 i10_7th Ave
        h,w = 750, 240
        lane = 5
        scene_mask = cv2.imread('scene_6_mask.bmp')
        plt.imshow(scene_mask)
        
        raw_video = cv2.imread('Test6_n_part1_Moment.jpg')
        plt.imshow(raw_video)
        
        GPSMap_raw = cv2.imread('scene_6.png')
        plt.imshow(GPSMap_raw)
        
        rec_map = np.zeros((h, w))
        # plt.imshow(GPSMap)

        # Key points from video
        # Test 1
        keypoints_location = [(358, 777),
                              (1179, 368), 
                              (1548, 406),
                              (1231, 978)]
        
        # GPS view
        # Test 1
        keypoints_location_GPS = [(1059, 1288),
                                  (965, 539), 
                                  (1212, 515),
                                  (1306, 1250)]
        
        # Project to satellite view
        keypoints_location_ref = [(0, h),
                                  (0, 0), 
                                  (w, 0),
                                  (w, h)]
        
        # Time frame list
        time_list =  [datetime(2022, 1, 13,  7, 56, 5),
                      datetime(2022, 1, 13,  9, 56, 6),
                      datetime(2022, 1, 13, 11, 56, 7),
                      datetime(2022, 1, 13, 13, 56, 8),
                      datetime(2022, 1, 13, 15, 56, 9)]

        
        
    if scene_id == 8:
        # Test 1 i10_7th Ave
        h,w = 1200, 320
        lane = 6
        scene_mask = cv2.imread('scene_8_mask.bmp')
        plt.imshow(scene_mask)
        
        raw_video = cv2.imread('Test8_n_part3_Moment.jpg')
        plt.imshow(raw_video)
        
        GPSMap_raw = cv2.imread('scene_8.png')
        plt.imshow(GPSMap_raw)
        
        rec_map = np.zeros((h, w))
        # plt.imshow(GPSMap)

        # Key points from video
        # Test 1
        keypoints_location = [(343, 635),
                              (1437, 283), 
                              (1786, 300),
                              (1443, 896)]
        
        # GPS view
        # Test 1
        keypoints_location_GPS = [(628, 1529),
                                  (532, 184), 
                                  (872, 160),
                                  (983, 1486)]
        
        # Project to satellite view
        keypoints_location_ref = [(0, h),
                                  (0, 0), 
                                  (w, 0),
                                  (w, h)]
        
        # Time frame list
        time_list =  [datetime(2022, 1, 5,   8, 5, 48),
                      datetime(2022, 1, 5,  10, 5, 49),
                      datetime(2022, 1, 5,  12, 5, 50),
                      datetime(2022, 1, 5,  14, 5, 52),
                      datetime(2022, 1, 5,  16, 5, 53)]
        
        
        
        
        
    if scene_id == 9:
        # Test 9 i10_7th Ave
        h,w = 650, 440
        lane = 8
        scene_mask = cv2.imread('scene_9_mask.bmp')
        plt.imshow(scene_mask)
        
        raw_video = cv2.imread('Test9_n_part3_Moment.jpg')
        plt.imshow(raw_video)
        
        GPSMap_raw = cv2.imread('scene_9.png')
        plt.imshow(GPSMap_raw)
        
        rec_map = np.zeros((h, w))
        # plt.imshow(GPSMap)
        
        keypoints_location = [(744, 990),(300, 475),
                              (940, 400),(1782, 663)]
        
        # GPS view
        keypoints_location_GPS = [(278, 1270),(265, 805),
                                  (590, 805),(603, 1270)]
        
        # Project to satellite view
        keypoints_location_ref = [(0, h),(0, 0),
                                  (w, 0),(w, h)]   
        
        # Time frame list
        time_list =  [datetime(2022, 1, 6,  7, 29, 32),
                      datetime(2022, 1, 6,  9, 29, 33),
                      datetime(2022, 1, 6, 11, 29, 34),
                      datetime(2022, 1, 6, 13, 29, 36),
                      datetime(2022, 1, 6, 15, 29, 37)]


    data = scene_mask.reshape((-1,3))
    data = np.float32(data)
    K = lane
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    ret,label,center = cv2.kmeans(data,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    dst = res.reshape((scene_mask.shape))
    plt.imshow(dst)
    
    mask = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    plt.imshow(mask)
    
    lane = np.unique(mask)
    for i in range(len(lane)):
        mask[mask==lane[i]] = i+1
        
    return [raw_video, GPSMap_raw, rec_map, h, w, 
            keypoints_location, keypoints_location_ref, keypoints_location_GPS, 
            time_list, lane, mask]








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






def get_time(dataframe, time):
    '''
    Get time information in real time coordinate
    '''
    
    init_time = time
    print(init_time)
    
    for i in range(len(dataframe)):
        
        frame = dataframe[i, index_Frame]
        seconds = frame//30
        target_time = init_time + timedelta(seconds=seconds)

        dataframe[i, index_Time] = target_time
        
        # if i%100 == 0: print(target_time, i)
        
    return dataframe



    
    


def IVVR_OVVR_OSR(dataframe, speed_limit=65):
    speed_limit = speed_limit
    
    count = 0
    Vi = []
    Vi_mean = []
    I_1 = 0
    I_2 = 0
    I_3 = 0
    
    target_list = np.unique(dataframe[:, index_Target])
    length = len(target_list)
    
    for i in target_list:
        target_i_index = get_target_i(dataframe, i)[1]
        target_i       = get_target_i(dataframe, i)[0]
    
        # IVVR
        vi_max = max(target_i[:,index_Instantaneous_V])
        vi_min = min(target_i[:,index_Instantaneous_V])
        vi_mean = target_i[0,index_Mean_V]
        if vi_mean == 0:
            Vi.append(0)
        else:
            Vi.append(abs(vi_max-vi_min)/vi_mean)
        
        # OVVR
        Vi_mean.append(vi_mean)
        
        # OSR
        if vi_max > speed_limit:
            I_1 = I_1+1
            
        # OSR+
        if vi_max > speed_limit+10:
            I_2 = I_2+1
            
        # OSR++
        if vi_max > speed_limit+20:
            I_3 = I_3+1        
        
        
        if count%100 == 0:
            print(str(count) + '/' + str(length))
            
        count = count + 1
    
    
    N = len(np.unique(dataframe[:, index_Target]))
    
    # IVVR
    IVVR = np.sum(np.array(Vi))/N
    
    # OVVR
    Vi_mean = np.array(Vi_mean)
    Vmean = np.sum(Vi_mean)/N
    OVVR = np.sum(abs(Vi_mean-Vmean)/Vmean)/N

    # OSR
    OSR = I_1/N
    OSR_p = I_2/N
    OSR_pp = I_3/N

    
    
    # VOL
    VOL = len(target_list)

    return IVVR,OVVR,OSR,OSR_p,OSR_pp,VOL









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
        content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(int)
    
    return content
    

# trucks_1 = txt_to_dataframe(file_path+'Test1_14_1_5.txt')    
# trucks_2 = txt_to_dataframe(file_path+'Test1_14_1_7.txt')    
# cars = txt_to_dataframe(file_path+'Test1_14_1_2.txt')    
    
# trucks = len(np.unique(trucks_1[:,1])) + len(np.unique(trucks_2[:,1]))
# cars = len(np.unique(cars[:,1]))


def TCI(trucks, cars):
    f1 = trucks / (trucks+cars)
    f2 = cars / (trucks+cars)

    TCI = 1/(2*(1-2*f1*f2))    

    return f1,f2,TCI


# trucks = 50
# cars = 10000
# TCI(trucks, cars)


def NTC(trucks, cars, scene_id, h, w):
    if scene_id == 1 or 4:
        Nl = 5
        
    if scene_id == 6 or 8:
        Nl = 6

    if scene_id == 9:
        Nl = 8


    
    l1 = 16
    l2 = 4.5
    
    # l_mean = (trucks*l1 + cars*l2) / (trucks+cars)
    l_mean = (trucks*l1 + cars*l2)
    
    # Nl = 5
    L = h
    
    NTC = l_mean / (Nl*L)
        
    return NTC







# dataframe = Traffic_info


def get_time_i(dataframe, start_time, end_time):
    '''
    Get target i from all frames
    
    '''
    dataframe_index = np.where((dataframe[:,index_Time]>=start_time) & (dataframe[:,index_Time]<=end_time))[0]
    dataframe_i       = dataframe[ dataframe_index ]
    
    return dataframe_i, dataframe_index




# dataframe = copy.deepcopy(Traffic_i)

'''
    extract TTC-CV
'''
def get_TTC(dataframe, h, w, dense_unit=60):
    FRAME = np.unique(dataframe[:,index_Frame])
    
    TTC_FRAME = np.zeros((len(FRAME),2))
    TTC_FRAME[:,0] = FRAME
    
    TTC = [[],[]]
    
    # dense_map = np.zeros((h,w), dtype=np.uint8)
    # dense_unit = w//4 # 100 v12 for test 1
            
    for f in range(len(FRAME)): 
        if f%1000==0: print('%d|%d'%(f, len(FRAME)))

        if (f%30)==0: # print(f)

            frame,frame_list = get_frame_i(dataframe, FRAME[f])
            num_vehicle = len(frame)
            
            # if num_vehicle > 6: print(f)
            
            if num_vehicle == 0:
                continue
            
            
            # distance-based clusters
            dense_map = np.zeros((h,w), dtype=np.uint8)
            for n in range(frame.shape[0]):
                i = round(frame[n,4])
                j = round(frame[n,3])
                
                for x  in range(2*dense_unit+1):
                    for y in range(2*dense_unit+1):
                        if (x-dense_unit)**2+(y-dense_unit)**2<=dense_unit**2:
                            if i-dense_unit+x>0 and i-dense_unit+x<h and j-dense_unit+y>0 and j-dense_unit+y<w:
                                dense_map[i-dense_unit+x,j-dense_unit+y]= 255
                                
            # plt.imshow(dense_map)
            
            
            num_objects, labels = cv2.connectedComponents(dense_map)
            # if num_objects == 1:
            #     flag_1 = 1
                
            # if num_objects == 2:
            #     flag_2 = 1
            # Cat Cluster
            # 0. cluster
            # 1. x
            # 2. y
            # 3. V
            # 4. min_x
            # 5. max_x
            # 6. ttc
            cluster = np.zeros((num_objects-1,7), dtype=object)
            TTC_i = []

            if num_objects < 3:
                continue
                # TTC_i = 0
                # TTC[0].append(f)
                # TTC[1].append(TTC_i)
                
            else:
                # Get cluster
                for i in range(1, num_objects):
                    cluster[i-1,0] = i
                    
                    cluster_i_x = np.where(labels==i)[1]
                    cluster_i_y = np.where(labels==i)[0]
                    x_max = np.max(cluster_i_x)
                    x_min = np.min(cluster_i_x)
                    y_max = np.max(cluster_i_y)
                    y_min = np.min(cluster_i_y)
                    
                    # cluster_i_x = (x_max + x_min)//2   
                    # cluster_i_y = (y_max + y_min)//2
                    cluster_i_x = np.mean(cluster_i_x)
                    cluster_i_y = np.mean(cluster_i_y)
                    
                    #print(cluster_i_y)
                    cluster[i-1,1] = cluster_i_x
                    cluster[i-1,2] = cluster_i_y
                    
                    Vt_mean = []
                    for n in range(len(frame)):
                        if frame[n,4] > (y_min-1) and frame[n,4] < (y_max+1):
                            Vt_mean.append(frame[n,8])

                    cluster_Vmean = np.mean(Vt_mean)
                    cluster[i-1,3] = cluster_Vmean
                    
                    cluster[i-1,4] = x_min
                    cluster[i-1,5] = x_max
                    
                # max_y = 900 #THE potential collision point predefine for segment 2
                # Get TTC
                for i in range(len(cluster)):
                    
                    cluster_i = cluster[i]
                    # if cluster_i[2]>max_y:
                    #     break
                    
                    
                    # Caclute TTC for segment#1
                    cluster_sub = np.delete(cluster, i, 0)
                    
                    # dist = np.where(abs(cluster_sub[:,1]-cluster_i[1]) > dense_unit)[0] # horizon dist 
                    rule_1 = np.where(cluster_sub[:,4]>cluster_i[5])[0] # x_min
                    rule_2 = np.where(cluster_sub[:,5]<cluster_i[4])[0] # x_max

                    # dist = np.where(cluster_sub[:,2] > max_y )[0]
                    rule_3 = np.where(cluster_sub[:,2]-cluster_i[2]>  0)[0]   ## LOOK FORWARD CLUSTER
                    # print(dist)
                    # print(minus_dist)
                    dist = list(set(rule_1)|set(rule_2)|set(rule_3))
                    # print(dist)
                    
                    cluster_sub = np.delete(cluster_sub, dist, 0)
                    # print(len(cluster_sub))
                    # #cluster_sub = np.delete(cluster_sub, minus_dist, 0)
                    # print(len(cluster_sub))
                    if len(cluster_sub) == 0:
                        #TTC_i.append(0)   # 0?
                        continue
                    else:
                        dist_y = abs(cluster_sub[:,2] - cluster_i[2])
                        cluster_n = cluster_sub[dist_y==min(dist_y)][0]
                        ttc = (cluster_i[2]-cluster_n[2])/10 / (cluster_i[3]-cluster_n[3])

                        # ttc =  ((max_y-cluster_i[2])/10/cluster_i[3]) # ttc for segment 2 . the merge .
                        # if ttc<0:
                            #print(False)
                        # print(cluster_i[2], cluster_n[2], cluster_i[3], cluster_n[3], ttc)
    
                        # if ttc < np.inf and ttc >= 0:
                        #     TTC_i.append(ttc)
                        # else:
                        #     TTC_i.append(nan)
                        # print(ttc)
                        if ttc < np.inf and ttc >= 0:
                            # print(f, ttc)
                            TTC_i.append(ttc)

                            
                            
                if len(TTC_i)>0:   # get TTC-CV
                    # print(f,TTC_i)
                    # if len(TTC_i)==1: TTC_i = TTC_i[0]
                    # else:
                        
                    TTC_i = np.nanstd(TTC_i)/ np.nanmean(TTC_i)
                    TTC_i = TTC_i/((num_objects-1)/num_vehicle)
                    
                    # for t in range(len(TTC_i)):
                    if np.isnan(TTC_i):
                        TTC_i = 0
                        
                    TTC[0].append(f)
                    TTC[1].append(TTC_i)
            
                # else:
                #     print(TTC_i)
                    
                # if np.isnan(TTC_i) :
                #     TTC_i = 0
                # else:
                #     TTC_i = TTC_i/((num_objects-1)/num_vehicle)
                
                # TTC[0].append(f)
                # TTC[1].append(TTC_i)
                
        # if f%100==0: print(f,TTC[1][-1])


    #print(len(TTC[1]))
    TTC_final = np.nanmean(TTC[1]) #* 1000
    # TTC_final = np.nanstd(TTC[1])/ np.nanmean(TTC[1])
    print(TTC_final)

    return TTC_final

# TTC = get_TTC(Traffic_i)    
    



# dataframe = copy.deepcopy(Traffic_i)

'''
    extract TTC
'''
def get_individual_TTC(dataframe, lane):
    FRAME = np.unique(dataframe[:,index_Frame])
    
    TTC_FRAME = np.zeros((len(FRAME),2))
    TTC_FRAME[:,0] = FRAME
    
    TTC_frame = []
    TTC_total = []
    
    # dense_map = np.zeros((h,w), dtype=np.uint8)
    # dense_unit = w//4 # 100 v12 for test 1
            
    for f in range(len(FRAME)): 

        if (f%30)==0: # print(f)
            TTC_i = []

            frame,frame_list = get_frame_i(dataframe, FRAME[f])
            num_vehicle = len(frame)
            
            if num_vehicle == 0:
                continue
            
            # Each lane
            for i in range(1,len(lane)+1):
                lane_i = frame[frame[:,10]==i]
                if len(lane_i) == 0:
                    continue
                
                # Each car
                for j in range(len(lane_i)):
                    car_j = lane_i[j]
                    y = car_j[4]
                    v = car_j[8]
                    other_cars = np.delete(lane_i, j, 0)
                    
                    # Each ttc
                    for k in range(len(other_cars)):
                        if other_cars[k,4] < y:
                            if v > other_cars[k,8]:
                                ttc_i = (y-other_cars[k,4])/10 / (v-other_cars[k,8])

                                TTC_i.append(ttc_i)                   
                                TTC_total.append(ttc_i)
                                
                    
            if len(TTC_i) > 0:
                TTC_frame.append(np.nanmean(TTC_i))
                # TTC_mean.append(np.nanstd(TTC_i)/ np.nanmean(TTC_i))
                # if np.isnan(TTC_mean):
                #     TTC_mean = 0
                        
    TTC_frame = np.nanstd(TTC_frame)/ np.nanmean(TTC_frame)
    TTC_total = np.sum(TTC_total)
    # TTC_total = np.nanmean(TTC[1]) #* 1000
    # TTC_final = np.nanstd(TTC[1])/ np.nanmean(TTC[1])
    # print(TTC_final)

    return TTC_frame, TTC_total

# TTC = get_TTC(Traffic_i)    
    


# dataframe = copy.deepcopy(Traffic_info)

def time_slice(dataframe, scene_id, frag_id):
    delet_list = []

    if scene_id == 1:
        if (frag_id+1) == 1:
            [start, end] = [datetime(2021, 11, 20, 10, 20, 0), datetime(2021, 11, 20, 10, 50, 0)]
            
        elif (frag_id+1) == 7:
            [start, end] = [datetime(2021, 11, 20, 17, 20, 0), datetime(2021, 11, 20, 17, 30, 0)]
        
        elif (frag_id+1) == 8:
            [start, end] = [datetime(2021, 11, 20, 17, 20, 0), datetime(2021, 11, 20, 18, 30, 0)]
            
        else: 
            return dataframe
        
    elif scene_id == 5:
        if (frag_id+1) == 1:
            [start, end] = [datetime(2021, 10, 25,  9, 30, 0), datetime(2021, 10, 25, 11, 30, 0)]
            
        elif (frag_id+1) == 2:
            [start, end] = [datetime(2021, 10, 25, 13, 0, 0), datetime(2021, 10, 25, 13, 40, 0)]
        
        elif (frag_id+1) == 3:
            [start, end] = [datetime(2021, 10, 25, 13, 30, 0), datetime(2021, 10, 25, 15, 10, 0)]
        else: 
            return dataframe

        
    elif scene_id == 8:
        if (frag_id+1) == 5:
            [start, end] = [datetime(2022, 1, 5,  17, 10, 0), datetime(2022, 1, 5,  18, 10, 0)]
        else: 
            return dataframe
        
    else: 
        return dataframe
    
    n = len(dataframe)
    for i in range(n):
        if start < dataframe[i,0] < end:
            delet_list.append(i)        
            
    # delet_list = [i for k in delet_list for i in k]
    dataframe = np.delete(dataframe, delet_list[::], axis=0)
        
    return dataframe

    



if __name__ == '__main__':
    '''
    scene_id = [1,4,5,6,8,9]
    '''
    
    for scene_id in [5,6,8,9]:

        # scene_id = 4
        
        
        File_List = glob.glob('./save/trajectory/Test'+str(scene_id)+'_*')
        Class_List = glob.glob('./save/class/Test'+str(scene_id)+'_*')
    
        [raw_video, GPSMap_raw, GPSMap, h, w, 
         keypoints_location, 
         keypoints_location_ref, 
         keypoints_location_GPS,
         time_list, lane, mask] = scene(scene_id)
    
        # File_path = './save/traffic/'
        # File_List = os.listdir(File_path)
        
        # Class_path = './save/class/'
        # Class_List = os.listdir(Class_path)
        
        
        # Total length: 10 Hours
        # Interval = 10 min
        # Total cell: 10*6 = 60
        
        # Traffic metrics
        Traffic = np.zeros((150 ,20),dtype=object)
        
        '''
        Time   = 0
        
        IVVR   = 1
        OVVR   = 2
        
        OSR    = 3
        OSR+   = 4
        OSR++  = 5
        
        VOL    = 6
        TCI    = 7
        NTC    = 8
        
        TTC_CV      = 9
        TTC_mean    = 10
        TTC_taotal  = 11

        
        Truck density = 13
        
        Name   = last col
        '''
        
        count = 0
        # hours = 0
        
        for i in range(len(File_List)): #14,19
            file = File_List[i]
            Class = Class_List[i]
            Traffic_info = np.load(file,  allow_pickle=True)
            Class_info   = np.load(Class, allow_pickle=True)
            
            trajectory_file = file[18:]
            class_file      = Class[13:]
        
            # time = time_list[hours]
            # hours = hours+1
            print(file)
            print(Class)
            
            Class_info = np.array(Class_info, dtype=object)
            Class_info = get_time(Class_info, time_list[i])
    
            #
            Traffic_info = time_slice(Traffic_info, scene_id, i)
            # Traffic_info[:,8] = Traffic_info[:,8]*2
        
            
            for minute in [10,20,30,40,50,60,70,80,90,100,110,120]:
                init_time = Traffic_info[0,0] + timedelta(minutes=(minute-10))
                
                # Time label
                start_time = init_time
                end_time   = init_time+timedelta(minutes=10)
                
                #
                Traffic_i = get_time_i(Traffic_info, start_time, end_time)[0]
                Class_i   = get_time_i(Class_info  , start_time, end_time)[0]
                
                if len(Traffic_i) == 0:
                    continue
                
                Traffic[count, 0] = start_time
                # Traffic[count, 8] = end_time
                Traffic[count, -1] = trajectory_file.replace('.npy','')
                
                
                # TCI
                trucks = len(np.where(Class_i[:,5]==7)[0])
                cars = len(np.where(Class_i[:,5]==2)[0])
                f1,f2,tci = TCI(trucks, cars)
                Traffic[count, 7] = tci
                # Traffic[count, 11] = trucks
                # Traffic[count, 12] = cars
                Traffic[count, 13] = f1


                # NTC
                Traffic[count, 8] = NTC(trucks, cars, scene_id, h, w)
        
                # IVVR OVVR OSR VOL
                [IVVR,OVVR,OSR,OSR_p,OSR_pp,VOL] = IVVR_OVVR_OSR(Traffic_i, 65)
                Traffic[count, 1:7] = [IVVR,OVVR,OSR,OSR_p,OSR_pp,VOL]
                
                # TTC-CV
                Traffic[count, 9] = get_TTC(Traffic_i, h, w, dense_unit=60) 
                
                
                
                # TTC-individual
                TTC_mean, TTC_total = get_individual_TTC(Traffic_i, lane)
                Traffic[count, 10] = TTC_mean
                Traffic[count, 11] = TTC_total

                
                
                print(start_time)
                count = count+1
        
        
        save_path = './'
        file_name = 'Test_'+str(scene_id)
        finalfile_npy = 'Metrics_' + file_name + '.npy'
        finalfile_csv = 'Metrics_' + file_name + '.csv'
        
        #
        # Traffic_old = np.load(finalfile_npy,allow_pickle=True)
        # Traffic[:,6] = Traffic_old[:,6]
        # Traffic_old[:,3] = Traffic[:,3]
        # Traffic = Traffic_old
        
        # Traffic_new = np.hstack((Traffic_old,Traffic[:,11:14]))
        # Traffic = Traffic_new
        
        
        np.save(save_path+finalfile_npy, Traffic)
        np.savetxt(save_path+finalfile_csv, Traffic, fmt='%s', delimiter=',')
        



# temp = copy.deepcopy(Traffic)

# temp = np.delete(temp, [13,14],0)




# plt.scatter(range(len(Traffic_info)), Traffic_info[:,7])

# plt.scatter(range(len(np.unique(Traffic_info[:,7]))), np.unique(Traffic_info[:,7]))



# # hour_list_total = []
# # for j in range(23):
# #     df_month = df.loc[df["IncidentHour"]==j]
# #     hour_list_total.append(df_month.shape[0])





# # plt.plot(hour_list_total[9:21])




# # file = file_path+'Test1_'+str(hours)+'_'+str(minutes)+'_All.txt'


# a = Traffic[0:31,0]
# b = Traffic[0:31,7]

# a_norm = a#(a-np.mean(a))/np.std(a)
# b_norm = b

# b_norm[:,0] = (b[:,0]-np.mean(b[:,0]))/np.std(b[:,0])
# b_norm[:,1] = (b[:,1]-np.mean(b[:,1]))/np.std(b[:,1])
# b_norm[:,2] = (b[:,2]-np.mean(b[:,2]))/np.std(b[:,2])


# # plt.scatter(a,b)
# # plt.plot(range(len(a_norm)),a_norm)
# # plt.plot(range(len(b_norm)),b_norm)

# b_norm = (b-np.mean(b))/np.std(b)


# plt.plot(a_norm, b_norm)
# plt.scatter(a_norm, b_norm)


# plt.plot(b)



# index = 73582
# b = get_target_i(dataframe, index)[0]
# plt.scatter(b[:,3], b[:,4])


# plt.plot(range(len(b)), b[:,8])
# plt.ylim(0, 120)



# np.where(Traffic_info[:,6] > 150)
# Traffic_info[845562]

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










