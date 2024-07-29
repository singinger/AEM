import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Normalize

import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
from promptCustom import prompt_tuning
import random, pdb, math, copy

from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

import clip
from PIL import Image
import time

def cross_entropy(y_hat,y):
    return -torch.log(y_hat ,y).mean()

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()  # source data txt
    txt_tar = open(args.t_dset_path).readlines()  # target data txt
    txt_test = open(args.test_dset_path).readlines()  # target data txt

    count = np.zeros(args.class_num)
    tr_txt = []
    te_txt = []
    for i in range(len(txt_src)):
        line = txt_src[i]
        reci = line.strip().split(' ')
        if count[int(reci[1])] < 1:
            count[int(reci[1])] += 1
            te_txt.append(line)
        else:
            tr_txt.append(line)
    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def cal_acc(loader, netF=None, netB=None, netC=None, netC1=None, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if netB is None:
                outputs = netC(netF(inputs))
                outputs_clip = netC1(netF(inputs))
            else:
                outputs = netC(netB(netF(inputs)))
                outputs_clip = netC1(netB(netF(inputs)))

            if start_test:
                all_output = outputs.float().cpu()
                all_output_clip = outputs_clip.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_output_clip = torch.cat((all_output_clip, outputs_clip.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    all_output_clip = nn.Softmax(dim=1)(all_output_clip)
    all_output_mix = nn.Softmax(dim=1)((all_output+all_output_clip)/2)

    _, predict = torch.max(all_output, 1)
    _, predict_clip = torch.max(all_output_clip, 1)
    _, predict_mix = torch.max(all_output_mix, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy_clip = torch.sum(torch.squeeze(predict_clip).float() == all_label).item() / float(all_label.size()[0])
    accuracy_mix = torch.sum(torch.squeeze(predict_mix).float() == all_label).item() / float(all_label.size()[0])

    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])
    mean_ent_clip = torch.mean(loss.Entropy(all_output_clip)).cpu().data.item() / np.log(all_label.size()[0])
    mean_ent_mix = torch.mean(loss.Entropy(all_output_mix)).cpu().data.item() / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
    
        matrix = confusion_matrix(all_label, torch.squeeze(predict_clip).float())
        matrix = matrix[np.unique(all_label).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc_clip = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc_clip = ' '.join(aa)
            
        matrix = confusion_matrix(all_label, torch.squeeze(predict_mix).float())
        matrix = matrix[np.unique(all_label).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc_mix = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc_mix = ' '.join(aa)
        return aacc, acc, aacc_clip, acc_clip, aacc_mix, acc_mix, mean_ent, mean_ent_clip, mean_ent_mix
    else:
        return accuracy * 100, accuracy_clip*100, accuracy_mix*100, mean_ent, mean_ent_clip, mean_ent_mix



def train_source_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    param_group = []
    learning_rate = args.lr_src
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // args.max_epoch
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        if args.dset == 'VisDA-2017':
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=4.5)
        else:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netF(inputs_source))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source,
                                                                                           labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, None, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netC


def test_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netC.eval()

    acc, _,_,_,_, _ = cal_acc(dset_loaders['test'], netF, None, netC, netC,False)
    log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

"""return the pseudo-labels"""
def fs_target_simp(args):
    
    dset_loaders = data_load(args)

    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir_src + '/source_C.pt'
    print(args.modelpath)
    netC.load_state_dict(torch.load(args.modelpath))

    source_model = nn.Sequential(netF,netB, netC).cuda()
    source_model.eval()

    start_test = True
    label_type = "soft"
    with torch.no_grad():
        iter_target = iter(dset_loaders["target_te"])
        for i in range(len(dset_loaders["target_te"])):
            data = iter_target.next()
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            # source_output = {Tensor:(64,31)}
            source_outputs = source_model(inputs)
            source_outputs = nn.Softmax(dim=1)(source_outputs)

            if label_type == "soft":
                if start_test:
                    all_output = source_outputs.float()
                    all_label = labels
                    start_test = False
                else:
                    all_output = torch.cat((all_output, source_outputs.float()), 0)
                    all_label = torch.cat((all_label, labels), 0)

        black_pred = all_output.detach()
  
    return black_pred



def get_text(categories, device):
    # (31,77) 
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in categories]).to(device)
    return text_inputs

def discrepancy(out1, out2):
    return -torch.mean(torch.sum(out1.softmax(dim=1)*out2.softmax(dim=1), dim=1))


def update_label(args, model, dataloader):
    model.eval()

    start_test = True
    args.load_prompt = osp.join(args.output_dir, "prompt_model.pt")

    with torch.no_grad():
        iter_test = iter(dataloader)
        for i in range(len(dataloader)):
            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()

            outputs = model(inputs)

            if start_test:
                all_output = outputs.float()
                start_test = False

            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
    model.train()



def clip_single_simp(args):
    dset_loaders = data_load(args)

    max_acc = 0
    start_test = True

    max_iter = args.max_epoch * len(dset_loaders["target"])

    black_pred = fs_target_simp(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.arch, device=device)
    text = get_text(args.categories, device)
    text_fea = clip_model.encode_text(text)
    text_fea = text_fea/text_fea.norm(dim=1, keepdim=True)

    is_net_B = True


    param_group = []
    learning_rate = args.lr_src

    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, pretrain=True).cuda()

    if is_net_B == True:
        netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                    bottleneck_dim=args.bottleneck).cuda()
        netC = network.feat_classifier(class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
        netC_clip = network.feat_classifier(class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    elif is_net_B == False:
        netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()
        netC_clip = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]

    if is_net_B == True:
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]

    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC_clip.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    netF = nn.Sequential(netF, netB).cuda()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    ent_best = 1.0

    interval_iter = max_iter // args.max_epoch
    iter_num = 0


    while iter_num < max_iter:
        
        start_time_clip = 0
        
        if args.ema < 1.0 and iter_num > 0 and iter_num % interval_iter == 0:
            netF.eval()
            netC.eval()
            netC_clip.eval()

            start_test = True
            
            if iter_num % (interval_iter) == 0 and iter_num >= args.clip_update_bengin:
                text_fea = prompt_tuning.main_worker(args, dset_loaders["target"], black_pred, iter_num=iter_num, max_iter=max_iter)
                args.load_prompt = osp.join(args.output_dir, "prompt_model.pt")
            with torch.no_grad():
                iter_test = iter(dset_loaders["target_te"])
                for i in range(len(dset_loaders["target_te"])):
                    data = iter_test.next()
                    inputs = data[0]
                    inputs = inputs.cuda()

                    outputs_1 = netC(netF(inputs))
                    outputs_2 = netC_clip(netF(inputs))

                    outputs_1 = nn.Softmax(dim=1)(outputs_1)
                    outputs_2 = nn.Softmax(dim=1)(outputs_2)
  
                    if start_test:
                        all_output_1 = outputs_1.float()
                        all_output_2 = outputs_2.float()
                        start_test = False

                    else:
                        all_output_1 = torch.cat((all_output_1, outputs_1.float()), 0)
                        all_output_2 = torch.cat((all_output_2, outputs_2.float()), 0)

                black_pred = black_pred * args.ema + all_output_1 * (1-args.ema)

            netF.train()
            netC.train()
            netC_clip.train()

        

        try:
            inputs_target, y, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx = iter_target.next()


        if inputs_target.size(0) == 1:
            continue
            
        iter_num += 1
        if args.dset == 'VisDA-2017':
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=2.25)
        else:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)
        inputs_target = inputs_target.cuda()

        with torch.no_grad():
            outputs_target_by_source = black_pred[tar_idx, :]
            
            image_fea = clip_model.encode_image(inputs_target)
            image_fea = image_fea/image_fea.norm(dim=1, keepdim=True)
            outputs_target_by_clip = image_fea @ text_fea.T

            outputs_target_by_source = nn.Softmax(dim=1)(outputs_target_by_source)
            outputs_target_by_clip = nn.Softmax(dim=1)(outputs_target_by_clip)

            one_hot_source = torch.argmax(outputs_target_by_source,dim=1)
            one_hot_clip = torch.argmax(outputs_target_by_clip,dim=1)


        optimizer.zero_grad()

        
        netF.train()
        if is_net_B == True:
            netB.train()
        netC.train()
        netC_clip.train()

        fea_target = netF(inputs_target)
        output_cls1 = netC(fea_target)
        output_cls2 = netC_clip(fea_target)
        probs_cls1 = nn.Softmax(dim=1)(output_cls1)
        probs_cls2 = nn.Softmax(dim=1)(output_cls2)

        crossentropyloss = nn.CrossEntropyLoss()
        loss_cls1 = crossentropyloss(output_cls1, one_hot_source)
        loss_cls2 = crossentropyloss(output_cls2, one_hot_clip)


        '''mix_up strategy'''
        alpha = 0.3
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(inputs_target.size()[0]).cuda()
        mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
        mixed_output1 = (lam * probs_cls1 + (1 - lam) * probs_cls1[index, :]).detach()
        mixed_output2 = (lam * probs_cls2 + (1 - lam) * probs_cls2[index, :]).detach()

        model = nn.Sequential(netF, netC)
        update_batch_stats(model, False)
        outputs_target_m = model(mixed_input)
        update_batch_stats(model, True)
        outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
        classifier_loss = args.mix*nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output1)


        loss_supervised = loss_cls1 + loss_cls2 + classifier_loss

        entropy_loss1 = torch.mean(loss.Entropy(probs_cls1))
        msoftmax1 = probs_cls1.mean(dim=0)
        gentropy_loss1 = torch.sum(- msoftmax1 * torch.log(msoftmax1 + 1e-5))
        entropy_loss1 -= gentropy_loss1
        loss_supervised += entropy_loss1 

        loss_supervised.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        for param in netF.parameters():
            param.requires_grad = False
        netF.eval()

        fea_target = netF(inputs_target)
        output_cls1 = netC(fea_target)
        output_cls2 = netC_clip(fea_target)

        loss_cls1 = crossentropyloss(output_cls1, one_hot_source)
        loss_cls2 = crossentropyloss(output_cls2, one_hot_clip)
        loss_supervised = loss_cls1 + loss_cls2
        loss_dis = discrepancy(output_cls1,output_cls2)

        loss1 = loss_supervised- loss_dis
        
        
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        for param in netF.parameters():
            param.requires_grad = True
     
        netF.train()
        netC.train()
        netC_clip.train()

        for param in netC.parameters():
            param.requires_grad = False      
        for param in netC_clip.parameters():
            param.requires_grad = False    

        for i in range(args.extractor_loop_num):
            fea_target = netF(inputs_target)
            output_cls1 = netC(fea_target)
            output_cls2 = netC_clip(fea_target)

            loss_dis = discrepancy(output_cls1, output_cls2)
            loss_dis.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        for param in netC.parameters():
            param.requires_grad = True      
        for param in netC_clip.parameters():
            param.requires_grad = True   

        
        netF.eval()
        netC.eval()
        netC_clip.eval()
        

        if iter_num % interval_iter == 0 or iter_num == max_iter:

            model_cls1 = nn.Sequential(netF,netB,netC).cuda()
            model_cls_clip = nn.Sequential(netF,netB,netC_clip).cuda()
            model_cls1.eval()
            model_cls_clip.eval()

            if not is_net_B:
                acc_s_te, acc_clip, acc_mix, mean_ent, mean_ent_clip, mean_ent_mix \
                    = cal_acc(loader=dset_loaders['test'], netF=netF, netC=netC, netC1=netC_clip,flag=False)
              
            elif is_net_B:
                acc_s_te, acc_clip, acc_mix, mean_ent, mean_ent_clip, mean_ent_mix \
                    = cal_acc(loader=dset_loaders['test'], netF=netF, netC=netC, netC1=netC_clip,flag=False)
                # acc_s_te1, mean_ent1 = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}, {:.2f}, {:.2f}%, Max = {:.2f}%, Ent = {:.4f}, {:.4f}, {:.4f}' \
                .format(args.name, iter_num, max_iter, acc_s_te, acc_clip, acc_mix, max_acc, mean_ent, mean_ent_clip, mean_ent_mix)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            
            if max_acc < max(acc_s_te, acc_clip, acc_mix):
                max_acc = max(acc_s_te, acc_clip, acc_mix)
                
                torch.save(netF.state_dict(), osp.join(args.output_dir, "source_F.pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "source_B.pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "source_C.pt"))
                torch.save(netC_clip.state_dict(), osp.join(args.output_dir, "source_C_clip.pt"))
            model_cls1.train()
            model_cls_clip.train()


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--max_epoch', type=int, default=70, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VisDA-2017', 'office31', 'image-clef', 'office-home', 'office-caltech', 'domain_net'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output', type=str, default='./ckpt/tar')
    parser.add_argument('--lr_src', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net_src', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output_src', type=str, default='./ckps_wB/src')

    parser.add_argument('--arch', type=str, default='ViT-B/32') 
    parser.add_argument('--ctx', type=str, default='a_photo_of_a') 
    parser.add_argument('--num_ctx', type=int, default=4) 
    parser.add_argument('--load_prompt', type=str, default=None)
    parser.add_argument('--tta_steps', type=int, default=1)

    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--extractor_loop_num', type=int,default=4,help="the iter number of update extractor")

    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--ema', type=float, default=0.9)
    parser.add_argument('--mix', type=float, default=1.0)
    parser.add_argument('--clip_update_bengin', type=int, default=0)

    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65

    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

    if args.dset == 'domain_net':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    
    if args.dset == 'VisDA-2017':
        names = ['train', 'validation']
        args.class_num = 12

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'

    args.output_dir_src = osp.join(args.output_src, args.dset,
                                    names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    if not args.distill:
        print(args.output_dir_src + '/source_F.pt')
        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source_simp(args)

        args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            if args.t == 'clipart':
                continue
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            test_target_simp(args)
    
    if args.distill:
        
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i      
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            time_log = time.strftime("%m-%d-%H:%M", time.localtime())

            args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
                                    names[args.s][0].upper() + names[args.t][0].upper(),time_log)
            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)


            args.out_file = open(osp.join(args.output_dir, 'log_tar.txt'), 'w')
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        
            args.class_path = folder + args.dset + '/' + 'category.txt'
            args.categories = []
            with open(args.class_path, 'r') as file:
                for line in file:
                    args.categories.append(line)

            print("clip start training")
            clip_single_simp(args)


