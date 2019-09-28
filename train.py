import pickle
import os
import time
import shutil
import torch
import data
from model import VSE
from evaluation import AverageMeter, LogCollector, eval_feat

import logging
import tensorboard_logger as tb_logger 

import argparse

def main():
    #训练外参数
    parser = argparse.ArgumentParser()   
    parser.add_argument('--data_type', default='rgb', help='path to datasets')
    parser.add_argument('--feature_path',default='',help='path to datasets')
    parser.add_argument('--anno_path',default='',help='path to datasets')
    parser.add_argument('--feature_prefix',default='',help='prefix of feaeture')
    parser.add_argument('--dropout',default=0.5,help='prefix of feature')
    parser.add_argument('--split_video_file',default='',help='prefix of feature')
    parser.add_argument('--num_pos_sample',default=10,help='prefix of feature')
    parser.add_argument('--num_neg_sample',default=10,help='prefix of feature')
    parser.add_argument('--num_epochs',default=30,type=int,help='Number of training epochs')
    parser.add_argument('--batch_size',default=128,type=int,help='Size of a training mini-batch')
    parser.add_argument('--embed_size',default=10240,typr=int,help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip',default=2,type=float,help='Gradient clipping threshold')
    parser.add_argument('--learning_rate',default=.001,type=float,help='Initial learning rate')
    parser.add_argument('--lr_update',default=15,type=int,help='Number of epochs to update the learning rate')
    parser.add_argument('--wrokers',default=10,type=int,help='Number of data loader workers')
    parser.add_argument('--log_step',default=10,type=int,help='Number of steps to print and record the log')
    parser.add_argument('--val_step',default=500,type=int,help='Number of steps to run validation')
    parser.add_argument('--logger_name',default='runs/runX',help='Path to save the model and Tensorboard log')
    parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to latest checkpoint default=none')
    parser.add_argument('--storage_place',default='',type=str,metavar='Path',help='path to latest checkpoint default=none')
    parser.add_argument('--instance_data_path',default='',type=str,metavar='PATH',
    help='path to the latest checkpoint')


    opt = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    #Load data loaders
    #1
    train_loader, val_loader = data.get_loaders(opt)
    
    #Construct the model
    #2
    model = VSE(opt)

    #optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            #Eiters is used to show logs as the continuation of another
            #training 
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(opt.resume,
            start_epoch,best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        #train for one epoch
        #3
        train(opt, train_loader, model, epoch)
        #evaluate on validation set
        rsum = 0

        #remeber best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')
    #4 val_loader (feat,id,labels)
    eval_feat(val_loader,model,opt.storage_place)

def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    #switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train_start()

        data_time.update(time.time() - end)

        model.logger = train_logger

        #update the model
        model.train_emb(*train_data)

        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch,i,len(train_loader),batch_time=batch_time,
                    data_time=data_time,e_log=str(model.logger)
                )
            )

        #record logs in tensorboard
        tb_logger.log_value('epoch',epoch,step=model.Eiters)
        tb_logger.log_value('step',i,step=model.Eiters)
        tb_logger.log_value('batch_time',batch_time.val,step=model.Eiters)
        tb_logger.log_value('data_time',data_time.val,step=model.Eiters)
        model.logger.tb_log(tb_logger,step=model.Eiters)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state,prefix+'_'+str(state['epoch'])+'_'+filename)
    if is_best:
        shutil.copyfile(prefix+'_'+str(state['epoch'])+'_'+filename,
        prefix+'model_best.pth.tar')


def adjust_learning_rate(opt,optimizer,epoch):

    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):#准确率计算函数
    """compute the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _,pred = output.topk(maxk,1,True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()