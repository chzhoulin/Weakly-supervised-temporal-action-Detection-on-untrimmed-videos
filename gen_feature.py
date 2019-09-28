import numpy as np 

def feature_concat(rgb_file, flow_file, feature_file):
    import scipy.io as sio
    data_rgb = sio.loadmat(rgb_file)
    data_flow = sio.loadmat(flow_file)
    feature_rgb = data_rgb['prediction_scores'].mean(1).squeeze()
    feature_flow = data_flow['prediction_scores'].mean(1).squeeze()

    print(feature_rgb.shape)
    print(feature_flow.shape)

    feature = np.concatenate((feature_rgb[0:feature_flow.shape[0],:],feature_flow), 
    axis=1)
    sio.savemat(feature_file,{'feature':feature})

feature_folder = '../evaluation_code/anet1.2_action_global_pool'
video_id = open('../evaluation_code/anet_1.2_train_id_list.txt','r')

for video in video_id:
    import os
    video = video.strip()
    rgb_feature_path = os.path.join(feature_folder, 'anet1.2_train_rgb_weak_tsn_bn_inception_average_seg3_attention_sw7_iter_10000_dense_test_'+video)
    flow_feature_path = os.path.join(feature_folder, 'anet1.2_train_flow_weak_tsn_bn_inception_average_seg3_attention_sw7_iter_10000_dense_test_'+video)
    feature_path = os.path.join(feature_folder, 'anet1.2_train_soft_untrimmednet_feature_dense_test_'+video)

    feature_concat(rgb_feature_path, flow_feature_path, feature_path)
    print('{} done'.format(video))
