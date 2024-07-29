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


# def prompt_main(args,dataloader,all_output, iter_num, model):
#     # This codebase has only been tested under the single GPU setting
#     # assert int(args.gpu_id) is not None
#     text_features, prompt_model = main_worker(args,dataloader,all_output, iter_num, model=model)
#     text_features = text_features.detach()
#     return text_features, prompt_model

def main_worker(args, dataloader):

    classnames = args.categories

    # model = get_cocoop(args.arch, classnames, int(args.gpu_id), args.num_ctx)
    model = get_coop(args.arch, args.dset, int(args.gpu_id), args.num_ctx, args.ctx)
    model = model.cuda()

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

    text_features = test_time_adapt_eval(dataloader, model, args)

    return text_features

def test_time_adapt_eval(dataloader, model, args):
    with torch.no_grad():
        model.train()

    iter_test = iter(dataloader)
    # 对每个图像样本，和其对应的标签
    for i in range(len(dataloader)):
        images, _ = next(iter_test)

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda(int(args.gpu_id), non_blocking=True)
        
        with torch.no_grad():
            model.train()

        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            model.eval()
            _,text_features = model(images)
            
    return text_features