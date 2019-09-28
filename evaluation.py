from __future__ import print_function
import os 
import pickle
import numpy as np  
import time

import torch
from model import Classifier
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """string representation for logging
        """
        if self.count == 0:
            return str(self.val)
        
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to avl"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self,k,v,n=0):
        #create a new meter if pre not
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v,n)

    def __str__(self):
        """concatenate the meters in one log line"""
        s = ''
        for i, (k,v) in enumerate(iter(self.meters.items())):
            if i>0:
                s+='  '
            s+=k+' '+str(v)
        
        return s

    def tb_log(self,tb_logger,prefix='',step=None):
        """Log using tensorboard"""
        for k,v in iter(self.meters.items()):
            tb_logger.log_value(prefix+k,v.val,step=step)

def eval_feat(data_loader,model,storage_place): #
    model.val_start()
    anchor_window = [16,32,64,128,256,512]
    for i,(feat,id,labels) in enumerate(data_loader):
        print(id[0]+'is done')
        output = torch.zeros(feat.shape[0],101).cuda()

        for window in anchor_window:
            for j in range(output.shape[0]):
                start = int(j - window // 2)
                if start<0:
                    start=0
                end_pos = int(j+window//2)
                if end_pos>=feat.shape[0]:
                    end_pos=feat.shape[0]-1
                
                feat_pick = []
                window_size = end_pos - start
                start_prev = int(start - window_size * 0.25)
                end_end = int(end_pos + window_size * 0.25)
                if start_prev < 0:
                    start_prev = 0
                if end_end >= feat.shape[0]:
                    end_end = feat.shape[0]-1

                pick_first = int((start_prev+start)//2)
                pick_middle_first = int((start+int(start+window_size/3))//2)
                pick_middle_middle = int((int(start+window_size/3)+int(start+
                window_size*2/3))//2)
                pick_middle_last = int((int(start+window_size*2/3)+end_pos)//2)
                pick_end = int((end_pos+end_end)//2)

                feat_this_sample = [feat[pick_first].unsqueeze(0),
                feat[pick_middle_first].unsqueeze(0),feat[pick_middle_middle].unsqueeze(0),
                feat[pick_middle_last].unsqueeze(0),feat[pick_end].unsqueeze(0)]

                feat_this_sample = torch.cat(feat_this_sample,1).cuda()
                with torch.no_grad():
                    #@
                    output_feat_this_sample = model.forward_emb(feat_this_sample)

                output[start:end_pos] = output[start:end_pos]+output_feat_this_sample/(
                end_end - start_prev)

        for label in labels:
            torch.save(output[:,label],os.path.join(storage_place,id[0]+'_'+
            str(label)+'_score'))       
                
def slide_window_test(data_loader,model,storage_place): 
    import numpy as np
    model.val_start()
    anchor_window = [16,32,64,128,256]
    for i, (feat, id, labels) in enumerate(data_loader):
        print(id[0]+'is done')
        output = np.zeros((feat.shape[0],101))
        average_output = np.zeros((feat.shape[0],101))

        for window in anchor_window:
            for j in range(output.shape[0]):
                start = int(j - window // 2)
                if start<0:
                    start=0
                end_pos = int(j+window//2)
                if end_pos>=feat.shape[0]:
                    end_pos=feat.shape[0]-1
                
                feat_pick = []
                window_size = end_pos - start
                gap = window_size * 0.25
                if gap > 10:
                    gap = 10
                
                start_prev = int(start - gap)
                end_end = int(end_pos + gap)
                if start_prev < 0:
                    start_prev = 0
                if end_end >= feat.shape[0]:
                    end_end = feat.shape[0]-1

                middle_start = int(start+window_size/3)
                middle_end = int(start+window_size*2/3)

                if start_prev == start:
                    feat1 = feat[start,:]
                else:
                    feat1 = feat[start_prev:start,:].mean(0)
                
                if start == middle_start:
                    feat2 = feat[start,:]
                else:
                    feat2 = feat[start:middle_start,:].mean(0)

                if middle_start == middle_end:
                    feat3 = feat[middle_start,:]
                else:
                    feat3 = feat[middle_start:middle_end,:].mean(0)

                if middle_end == end_pos:
                    feat4 = feat[middle_end,:]
                else:
                    feat4 = feat[middle_end:end_pos,:].mean(0)

                if end_pos == end_end:
                    feat5 = feat[end_pos-1,:]
                else:
                    feat5 = feat[end_pos:end_end,:].mean(0)

                feat_this_sample = [feat1.unsqueeze(0),feat2.unsqueeze(0),feat3.unsqueeze(0),
                feat4.unsqueeze(0),feat5.unsqueeze(0)]

                feat_this_sample = torch.cat(feat_this_sample,1).cuda()
                with torch.no_grad():
                    output_feat_this_sample = model.forward_emb(feat_this_sample)

                output[j,:] = output_feat_this_sample
                average_output[start:end_pos] = average_output[start:end_pos]+output_feat_this_sample

            import scipy.io as sio
            sio.savemat(os.path.join(storage_place,id[0]+'_window_'+str(window)+'.mat'),
            {'prediction_scores': output, 'average_score':average_output})

"""
        for label in labels:
            torch.save(output[:,label],os.path.join(storage_place,id[0]+'_'+
            str(label)+'_score'))    
            """   
