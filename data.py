import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np 
import scipy.io as sio

class PrecompDataset(data.Dataset):
    """Load precomputed captions and features
       Possible options:f8k, f30k, coco, 10crop
    """

    def __init__(self, split, feature_path, feature_prefix, anno_path,
    data_path,num_pos_sample,num_neg_sample, data_type):
        super(PrecompDataset, self).__init__()
        id_file = [id.strip() for id in open(data_path,'r')]
        self.id_file = id_file
        self.data_path = data_path
        self.split = split
        self.database = {}
        self.num_pos_sample = num_pos_sample
        self.num_neg_sample = num_neg_sample

        if os.path.exists(anno_path + '_database'):
            self.database = torch.load(anno_path + '_database')
        else:
            for id in id_file:
                pos_sample, neg_sample=[], []
                total_path = os.path.join(anno_path,id+'.anno')
                print(total_path)
                anno_file = open(total_path,'r')
                for line in anno_file:
                    line_split = line.split(' ')
                    start, end, score, label = int(line_split[0]), int(line_split[1]),int(line_split[2]),int(line_split[3])

                    if label < 100:
                        pos_sample.append([start,end,label])
                    else:
                        neg_sample.append([start,end,label])

                if (len(pos_sample) + len(neg_sample)) == 0:
                    print('{} is deleted'.format(id))
                else:
                    self.database[id] = {}
                    self.database[id]['pos_sample'] = np.array(pos_sample)
                    self.database[id]['neg_sample'] = np.array(neg_sample)
                    self.database[id]['pos_len'] = len(pos_sample)
                    self.database[id]['neg_len'] = len(neg_sample)
                    self.database[id]['feat_path'] = os.path.join(feature_path,
                    feature_prefix+'_'+id+'.mat')

            torch.save(self.database,anno_path + '_database')
        self.id_file = [id for id in self.database]
        self.length = len(self.id_file)


    def __getitem__(self,index):
        #handle the image redundancy
        id = self.id_file[index]
        feat = torch.tensor(sio.loadmat(self.database[id]['feat_path'])['feature'])
        #torch.randperm() n个数的随机排列
        pos_sample_id = torch.randperm(self.database[id]['pos_len'])[0:self.num_pos_sample]
        neg_sample_id = torch.randperm(self.database[id]['neg_len'])[0:self.num_neg_sample]
        feat_pos, label_pos = self.__pick_sample(feat,self.database[id]['pos_sample'],
        pos_sample_id)
        feat_neg, label_neg = self.__pick_sample(feat,self.database[id]['neg_sample'],
        neg_sample_id)

        feat_total = []
        feat_total.extend(feat_pos)
        feat_total.extend(feat_neg)
        label_total = []
        label_total.extend(label_pos)
        label_total.extend(label_neg)

        try:
            feat_total = torch.cat(feat_total,0)
            label_total = torch.tensor(np.array(label_total))
            return feat_total,label_total
        except:
            print(id)

    def __pick__sample(self, feat, database, sample_id):
        feat_subsample = []
        label_subsample = []
        for i in sample_id:
            start = int(database[i][0])
            end_pos = int(database[i][1])
            label = int(database[i][2])

            feat_pick = []
            window_size = end_pos - start + 1
            gap = window_size * 0.25
            if gap > 10:
                gap = 10
             
            start_prev = int(start - gap)
            end_end = int(end_pos + gap)

            if start_prev < 0:
                start_prev = 0
            if end_end > feat.shape[0]:
                end_end = feat.shape[0]
            
            if start_prev == start:
                pick_first = torch.LongTensor(1).random_(start_prev, start_prev + 1)
            else:
                pick_first = torch.LongTensor(1).random_(start_prev, start)
            
            if start == int(start+window_size/3):
                pick_middle_first = torch.LongTensor(1).random_(start, start+1))
            else:
                pick_middle_first = torch.LongTensor(1).random_(start, int(start + window_size/3))
            
            if int(start+window_size/3) == int(start+window_size*2/3):
                pick_middle_middle = torch.LongTensor(1).random_(int(start+window_size/3),
                int(start+window_size/3)+1)
            else:
                pick_middle_middle = torch.LongTensor(1).random_(int(start + window_size/3),
            int(start+window_size*2/3))

            if int(start+window_size*2/3) == end_pos:
                pick_middle_last = torch.LongTensor(1).random_(int(start+window_start*2/3),
                int(start+window_start*2/3)+1)
            else:
                pick_middle_last = torch.LongTensor(1).random_(int(start+window_size*2/3),
            end_pos)

            if end_pos == end_end:
                pick_end = torch.LongTensor(1).random_(end_pos,end_pos+1)
            else:
                pick_end = torch.LongTensor(1).random_(end_pos,end_end)

            feat_this_sample = [feat[pick_first],feat[pick_middle_first],
            feat[pick_middle_middle],feat[pick_middle_last],feat[pick_end]]

            feat_this_sample = torch.cat(feat_this_sample, 1)
            assert len(feat[pick_first].shape) < 3

            feat_subsample.append(feat_this_sample)
            label_subsample.append(label)

        return feat_subsample, label_subsample

    def __len__(self):
        return self.length

def collate_fn(data):
    """Build mini-batch tensors from a list of (iamge, caption) tuples.
        args:
            data: list of (image, caption) tuple.
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.
        returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
    """

    # sort a data list by caption length
    images, labels = zip(*data)

    #Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)

    return images, labels


class Eval_PrecompDataset(data.Dataset):
    """Load precomputed captions and image features
       Possible options: f8k, f30k, coco, 10crop
    """
    def __init__(self, feature_path, feature_prefix, instance_data_path):
        super(Eval_PrecompDataset,self).__init__()
        id_file = [[id.strip().split(' ')[0], int(id.strip().split(' ')[1])] for id 
        in open(instance_data_path,'r')]
        database = {}
        database_id_map = []
        for id, label in id_file:
            if id[-2] == '_':
                id = id[0:-2]
            else:
                id = id[0:-3]
            
            if id+'@@'+str(label) not in database:
                database[id+'@@'+str(label)] = {}
                database[id+'@@'+str(label)]['label'] = [label]
                database[id+'@@'+str(label)]['feat_path'] = os.path.join(feature_path,
                feature_prefix+'_'+id+'.mat')
                database_id_map.append(id+'@@'+str(label))

        self.database = database
        self.database_id_map = database_id_map
        self.length = len(database)

    def __getitem__(self, index):
        #handle the image redundancy
        id = self.database_id_map[index]
        feat = torch.tensor(sio.loadmat(self.database[id]['feat_path'])['feature'])
        label = torch.tensor(np.array(self.database[id]['label']))
        return feat, id.split('@@')[0], label

    def __len__(self):
        return self.length


def eval_collate_fn(data):
    #Sort a data list by caption length
    images, id, label =zip(*data)

    #Merge images
    images = torch.cat(images, 0)

    return images, id, label



def get_precomp_loader(split, feature_path, feature_prefix, anno_path, data_path,
num_pos_sample, num_neg_sample,data_type, batch_size=100, shuffle=True, num_workers=2,
instance_data_path = None):

    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    #anno pos/neg sample的map exact_feat/label
    dset = PrecompDataset(split, feature_path, feature_prefix, anno_path, data_path, 
    num_pos_sample, num_neg_sample,data_type)
    #instance_data
    dset_eval = Eval_PrecompDataset(feature_path,feature_prefix,instance_data_path)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    eval_data_loader = torch.utils.data.DataLoader(dataset=dset_eval,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=eval_collate_fn)

    return data_loader, eval_data_loader

#loader
def get_loaders(opt):
    train_loader, eval_loader = get_precomp_loader('train', opt.feature_path, 
    opt.feature_prefix, opt.anno_path, opt.split_video_file, opt.num_pos_sample,
    opt.num_neg_sample, opt.data_type, opt.batch_size, True, opt.workers, opt.instance_data_path)
    
    return train_loader, eval_loader
