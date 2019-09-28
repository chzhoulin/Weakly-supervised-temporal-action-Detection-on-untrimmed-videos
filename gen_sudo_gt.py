import numpy as np 
import torch

OUT_RANGE_CONSTANT = -100000000

def softmax(raw_score,T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def data_preprocess_fusion(file_rgb, file_flow, label, att_thres=7):
    import scipy.io as sio
    data_rgb = sio.loadmat(file_rgb)
    data_flow = sio.loadmat(file_flow)

    att_score_rgb = data_rgb['attention_scores'].mean(1).squeeze()
    att_score_flow = data_flow['attention_scores'].mean(1).squeeze()
    att_score = 0.5 * att_score_rgb[0:att_score_flow.shape[0]] + 0.5 * att_score_flow

    pred_score_rgb = data_rgb['prediction_scores'].mean(1).squeeze()
    pred_score_flow = data_flow['prediction_scores'].mean(1).squeeze()
    pred_score=softmax(0.5*pred_score_rgb[0:pred_score_flow.shape[0],:]+
    0.5*pred_score_flow[:,:])[:,label]

    if att_thres >= att_score.max():
        att_thres = att_score.max() * 0.8

    filter_att = np.where(att_score < att_thres)
    pred_score[filter_att] = 0

    integral_pred_score = pred_score.copy()
    for i in range(pred_score.shape[0]):
        integral_pred_score[i] = pred_score[0:i].sum(0)
    integral_pred_score = np.append(integral_pred_score, pred_score.sum(0))
    return att_score, pred_score, integral_pred_score

def data_preprocess(file, label):
    import scipy.io as sio
    data = sio.loadmat(file)

    att_score = data['attention_scores'].mean(1)[0]
    pred_score = data['prediction_scores'].mean(1)[:,label]
    filter_att = np.where(att_score<7)

    pred_score[filter_att] = 0
    integral_pred_score = pred_score.copy()
    for i in range(pred_score.shape[0]):
        integral_pred_score[i] = pred_score[0:i].sum(0)
    return att_score, pred_score, integral_pred_score

def oic(integral_score, window, center, alpha, max_gap=10):
    x1 = int(np.floor(center - window/2))
    x2 = int(np.floor(center + window/2))

    if x1 < 0:
        x1 = 0
    if x2 >= integral_score.shape[0]:
        x2 = integral_score.shape[0] - 1

    gap = window * alpha
    if gap > max_gap:
        gap = max_gap
    
    if x2-x1+1 == 0:
        import ipdb; ipdb.set_trace()
        print(center)
        print(window)
        print(x2)
        print(x1)

    X1 = int(np.floor(x1 - gap))
    X2 = int(np.floor(x2 + gap))

    if X1 < 0:
        X1=0
    if X2 >= integral_score.shape[0]:
        X2 = integral_score.shape[0]-1  

    within = integral_score[x2] - integral_score[x1]
    without = integral_score[X2] - integral_score[X1]

    if X1 == x1 and X2 == x2:
        score = within / (x2-x1+1)
    else:
        score = (without-within)/((X2-X1+1) - (x2-x1+1)) - within/(x2-x1+1)

    return score, x1, x2

def oic_new(integral_score, start, win_size, alpha, max_gap=10):
    x1 = start
    x2 = start + win_size

    gap = win_size * alpha
    if gap > max_gap:
        gap = max_gap
    
    X1 = int(np.floor(x1 - gap))
    X2 = int(np.floor(x2 + gap))

    if X1 < 0:
        X1 = 0
    if X2 >= integral_score.shape[0]:
        X2 = integral_score.shape[0] - 1

    within = integral_score[x2] - integral_score[x1]
    without = integral_score[X2] - integral_score[X1]

    if X2 - X1:
        score = -1 * within / (x2-x1)
    else:
        score = (without - within) / ((X2-X1)-(x2-x1)) - within / (x2-x1)

    return score, x1, x2

def sliding_window_oic(integral_score, min_win=8, max_win=256, alpha=0.25):
    total_score = []

    if integral_score.shape[0] <= 2:
        return [(-1*integral_score[0],(0, integral_score.shape[0]-1))]

    if max_win >= integral_score.shape[0] * 0.9:
        max_win = int(integral_score.shape[0] * 0.9)
    if max_win <= 1:
        max_win = 2
    if min_win > integral_score.shape[0] or min_win >= max_win:
        min_win = 1
    
    for win_size in range(min_win, max_win):
        for i in range(0, integral_score.shape[0] - win_size):
            sub_total, com_x1, com_x2 = oic_new(integral_score, i, win_size, alpha)
            comb_x1_x2_pair = (com_x1,com_x2)
            total_score.append((sub_total,comb_x1_x2_pair))

    return total_score


def oic_whole_video(rgb_integral_score, flow_integral_score, window_size, float_scale,
    alpha):
    total_score = []

    combine_integral_score = (rgb_integral_score[0:flow_integral_score.shape[0]] + 
    flow_integral_score)/2
    if combine_integral_score.shape[0] < 16:
        window_size = [1,2,4,8]
    
    for anc_i, anchor_window in enumerate(window_size):
        move_in = int(-anchor_window*float_scale // 2)
        move_out = int(anchor_window*float_scale // 2 + 1)
        for win_i, variable_window in enumerate(range(move_in,move_out)):
            for i in range(combine_integral_score.shape[0]):
                actual_window = anchor_window + variable_window * 2
                if actual_window < 1:
                    actual_window = 1
                sub_total, comb_x1, comb_x2 = oic(combine_integral_score,actual_window,
                i,alpha)

                comb_x1_x2_pair = (comb_x1,comb_x2)
                if sub_total > OUT_RANGE_CONSTANT:
                    total_score.append((sub_total,comb_x1_x2_pair))
    return total_score

def bb_intersection_over_union(boxA,boxB):
    xA = max(boxA[0],boxB[0])
    xB = min(boxA[1],boxB[1])

    interArea = max(0,xB-xA+1)

    boxAArea = (boxA[1]-boxA[0]+1)
    boxBArea = (boxB[1]-boxB[0]+1)

    iou = interArea/float(boxAArea+boxBArea-interArea)

    return iou

#非极大值抑制
def nms(score, score_thres,iou_thres):
    i = 0
    score_negative = []
    tmp = min([s[0] for s in score])
    positive_thre = -1 * score_thres
    if tmp > positive_thre:
        positive_thre = tmp
    while i < len(score):
        if score[i][0] > positive_thre:
            if score[i][0]>score_thres:
                score_negative.append(score[i])
            del score[i]
        else:
            i = i+1
    i = 0

    while i<len(score):
        kickout = []
        window_i = score[i][1]

        for j in range(i+1,len(score)):
            window_j = score[j][1]
            iou = bb_intersection_over_union(window_i,window_j)
            if iou > iou_thres:
                kickout.append(j)

        for index in sorted(kickout, reverse=True):
            del score[index]
        i = i+1
    return score,score_negative

def gen_anno(score, thres, save_filename, label, score_thres):

    file = open(save_filename, 'a')
    score = list(set(score))
    place_score_list_ordered = sorted(score, key=lambda score_list: score_list[0])
    place_score_list_nms, place_score_list_nms_neg = nms(place_score_list_ordered,0.1,thres)
    for item in place_score_list_nms:
        score = item[0]
        start_frame = item[1][0]
        end_frame = item[1][1]
        file.write('{} {} {} {}\n'.format(start_frame,end_frame,score,label))

    for item in place_score_list_nms_neg:
        score = item[0]
        start_frame = item[0][1]
        end_frame = item[1][1]
        file.write('{} {} {} 100\n'.format(start_frame,end_frame,score))

    file.close()


def mp_jobs(video_label_combine):
    video = video_label_combine[0]
    label = video_label_combine[1]

    import os
    if os.path.exists(os.path.join(save_folder,'{}.anno'.format(video))):
        return
    else:
        rgb_video_fullname = os.path.join(score_folder,
        'anet1.2_train_rgb_weak_tsn_bn_inception_average_seg3_attention_sw7_iter_10000_dense_test_'+video)
        flow_video_fullname = os.path.join(score_folder,
        'anet1.2_train_flow_weak_tsn_bn_inception_average_seg3_attention_sw7_iter_18000_dense_test_'+video)

        att_score,pred_score,integral_score = data_preprocess_fusion(rgb_video_fullname,
        flow_video_fullname,label)
        
        #total_score.append((sub_total,comb_x1_x2_pair))
        comb_oic_score = sliding_window_oic(integral_score, 16, 512, alpha=0.25)
        gen_anno(comb_oic_score, thres = 0.9, save_filename = os.path.join(save_folder,
        '{}.anno'.format(video)), label = label, score_thres = 0.4)
        print('{} done'.format(video))

import os
import argparse
import multiprocessing

train_id_file = open('../evaluation_code/anet_1.2_train_instance_list.txt','r')
train_id_dict = {}
train_id_list = []

for line in train_id_file:
    line = line.strip()
    video = line.split(' ')[0][0:11]
    label = int(line.split(' ')[1])
    if video not in train_id_dict:
        train_id_dict[video] = [label]
    else:
        if label not in train_id_dict[video]:
            train_id_dict[video].append(label)

train_id_list =  [(video, label) for video in train_id_dict for label in train_id_dict[video]]

print(len(train_id_list))

window_size = [16,32,64,128,256,512]
score_folder = '../evaluation_code/anet1.2_action_score'
save_folder = 'anno_softmax_tmp'
#
#mp_jobs(train_id_list[1])

#for video in train_id_list:
    #mp_jobs(video)

pool = multiprocessing.Pool(48)
pool.map(mp_jobs,train_id_list)










