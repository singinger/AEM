from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn

from .iid_loss import IID_loss
import os.path as osp

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .cocoop import get_cocoop
from .custom_clip import get_coop
# from data.datautils_domain import  build_dataset
# from data.cls_to_names import *
# from data.domain_datasets import domain_datasets

def discrepancy(out1, out2):
    # return torch.mean(torch.abs(nn.functional.softmax(out1, dim=1) - nn.functional.softmax(out2, dim=1)))
    return -torch.mean(torch.sum(out1*out2, dim=1))
    
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

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def test_time_tuning(model, inputs, pesu_label, optimizer, args):
    for j in range(args.tta_steps):
        output,_ = model(inputs)
        # print(output.size(), pesu_label.size()) 
        pesu_label = pesu_label.cuda()
        output = nn.Softmax(dim=1)(output)
        loss = discrepancy(output, pesu_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 

# def prompt_main(args,dataloader,all_output, iter_num, model):
#     # This codebase has only been tested under the single GPU setting
#     # assert int(args.gpu_id) is not None
#     text_features, prompt_model = main_worker(args,dataloader,all_output, iter_num, model=model)
#     text_features = text_features.detach()
#     return text_features, prompt_model

def main_worker(args, dataloader, all_output, iter_num=0, max_iter=0):

    classnames = args.categories

    # model = get_cocoop(args.arch, classnames, int(args.gpu_id), args.num_ctx)
    model = get_coop(args.arch, args.dset, 0, args.num_ctx, args.ctx)
    model = model.cuda()

    if args.load_prompt is not None:
        print("loading prompt")
        pretrained_ctx = torch.load(args.load_prompt)['ctx']
        assert pretrained_ctx.size()[0] == args.num_ctx
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        # print(name)
        if "prompt_learner" not in name:
            param.requires_grad_(False)


    model.reset_classnames(classnames, args.arch)
    trainable_param = model.prompt_learner.parameters()
    
    if args.dset == 'VisDA-2017':
        optimizer = torch.optim.SGD(trainable_param, args.lr_src*0.001, weight_decay=1e-3,momentum=0.9,nesterov=False)
    else:
        optimizer = torch.optim.SGD(trainable_param, args.lr_src, weight_decay=5e-4,momentum=0.9,nesterov=False)
    # 存个学习率， 方便以后来计算
    optimizer = op_copy(optimizer)
    iter_num -= args.clip_update_bengin
    if args.dset == 'VisDA-2017':
        lr_scheduler(optimizer, iter_num=int(iter_num), max_iter=max_iter, power=2.25)
    else:
        lr_scheduler(optimizer, iter_num=int(iter_num), max_iter=max_iter, power=1.5)
    # optim_state = deepcopy(optimizer.state_dict())
    print(optimizer.param_groups[0]['lr'])
    cudnn.benchmark = True

    text_features = test_time_adapt_eval(dataloader, model, optimizer, args, all_output)

    return text_features

def test_time_adapt_eval(dataloader, model, optimizer, args, all_output):
    with torch.no_grad():
        model.train()

    iter_test = iter(dataloader)
    # 对每个图像样本，和其对应的标签
    for i in range(len(dataloader)):
        images, _, tar_idx = next(iter_test)
        pesu_label = all_output[tar_idx]

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda(int(args.gpu_id), non_blocking=True)
        
        with torch.no_grad():
            model.train()

        # optimizer.load_state_dict(optim_state)
        test_time_tuning(model,images,pesu_label, optimizer, args)
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            model.eval()
            _,text_features = model(images)
            
    #torch.save(model.prompt_learner.state_dict(), "./prompt_model.pt")
    torch.save(model.prompt_learner.state_dict(), osp.join(args.output_dir, "prompt_model.pt"))
    return text_features