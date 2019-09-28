import numpy as np 
import json 
import scipy.io as sio
import os

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[...,None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    xB = min(boxA[1], boxB[1])

    interArea = max(0, xB - xA + 1)

    boxAArea = (boxA[1]-boxA[0]+1)
    boxBArea = (boxB[1]-boxB[0]+1)

    iou = interArea / float(boxAArea+boxBArea-interArea)

    return iou

def nms(score, iou_thres):
    i=0
    while i < len(score):
        kickout = []
        window_i = score[i][1]
        for j in range(i+1, len(score)):
            window_j = score[j][1]
            iou = bb_intersection_over_union(window_i, window_j)
            if iou > iou_thres:
                kickout.append(j)

        for index in sorted(kickout, reverse=True):
            del score[index]
        i = i + 1
    
    return score

def get_detection(video_id, duration, label, score_path, window=[8,16,32,64,128,256],
thres=0.5):
    detections = []
    for win in window:
        data = sio.loadmat(os.path.join(score_path, '{}_window_{}.mat'.format(
            video_id, win)))
        scores = data['prediction_score']
        scores = softmax(scores)[:,label]

        for i in range(scores.shape[0]):
            if scores[i] >= thres:
                start = i-int(win*0.5)
                last = i+int(win*0.5)
                if start < 0:
                    start = 0
                if last >= scores.shape[0]:
                    last = scores.shape[0]-1
                detections.append((scores[i],(start,last)))

    detections_ordered = sorted(detections, key=lambda detection: detection[0], reverse=True)
    detections_frame = nms(detections_ordered, 0.5)

    step = duration / (scores.shape[0]-1)
    detections_time = []
    for detection in detections_frame:
        detections_time.append((detection[0],detection[1][0]*step+step*0.5, 
        detection[1][1]*step+step*0.5))

    return detections_time

file_json = '../evaluation_code/activity_net.v1-2.min.json'
fobj = open(file_json)
anet = json.load(fobj)
database = anet['database']

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

train_id_list = [(video, label) for video in train_id_dict for label in train_id_dict[video]]

for video in train_id_list:
    detections = get_detection(video[0], database[video[0]]['duration'], video[1], 
    'score_softmax_new')
    if len(detections) < 1:
        print('{} no detection'.format(video[0]))
    fid = open(os.path.join('detection_softmax_tmp','{}.det'.format(video[0])),'a')
    for detection in detections:
        fid.write('{} {} {} {}\n'.format(detection[0], detection[1], detection[2],
        video[1]))

    fid.close()
    print('{} done'.format(video[0]))












