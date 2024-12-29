import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
from jittor import transform
import os
import matplotlib.pyplot as plt
import jclip as clip
from default import HYPER_DICT
from copy import deepcopy
import argparse
import random
# 数据导入
train_label = pd.read_csv('../Dataset/train_1234.txt')
val_label = pd.read_csv('../Dataset/valid_b.txt')
test_label = pd.read_csv('../Dataset/test_b.txt')
class2target = pd.read_csv('../Dataset/class2target_b.txt')

train_label['path'] = '../Dataset/' + train_label['img_name']
val_label['path'] = '../Dataset/' + val_label['img_name']
test_label['path'] = '../Dataset/TestSetB/' + test_label['img_name']



# 手工模板
caltech101_templates = [
    'a photo of a {}'
]

food101_templates = [
    # 'a photo of a {}',
    'a photo of {}, a type of food'
]

animal_templates = [
    # 'a photo of a {}',
    'a photo of {}, a type of animal'
]
thu_dog_templates = [
    # 'a photo of a {}',
    'a photo of {}, a type of dog'
]

stanford_cars_templates = [
    'a photo of {}'
]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# 模型训练与验证
class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# 生产count个在[start, end之间随机数]
def generate_random_numbers(count, start=374, end=402):
    return [random.randint(start, end) for _ in range(count)]


# 文本特征提取
def extract_text_feature(target, text_encoder):

    # 因为一共有4个数据集，因此每个batch补充1/4 len(target) 个stanford-cars的text_feature
    # count = len(target) // 4
    # cars_label = [random.randint(374, 402) for _ in range(count)]
    # target.extend(cars_label)

    text_features = []
    text_labels = []
    for i in range(len(target)):
        label = target[i]
        # print(f'labele{i}-------{label}')
        cname = class2target[class2target['target'] == label]['class_name'].values[0]
        if cname.startswith('Animal'):
            cname = cname[7:]
            str_prompts = [template.format(cname.lower().replace("_", " ")) for template in animal_templates]
            labels = jt.array([label for _ in animal_templates])
        elif cname.startswith('Thu-dog'):
            cname = cname[8:]
            str_prompts = [template.format(cname.lower().replace("_", " ")) for template in thu_dog_templates]
            labels = jt.array([label for _ in thu_dog_templates])
        elif cname.startswith('Caltech-101'):
            cname = cname[12:]
            str_prompts = [template.format(cname.lower().replace("_", " ")) for template in caltech101_templates]
            labels = jt.array([label for _ in caltech101_templates])
        elif cname.startswith('Food-101'):
            cname = cname[9:]
            str_prompts = [template.format(cname.lower().replace("_", " ")) for template in food101_templates]
            labels = jt.array([label for _ in food101_templates])
        elif cname.startswith('Stanford-Cars'):
            cname = cname[14:]
            # print(cname)
            str_prompts = [template.format(cname.lower().replace("_", " ")) for template in stanford_cars_templates]
            labels = jt.array([label for _ in stanford_cars_templates])
        prompts = jt.cat([clip.tokenize(p) for p in str_prompts])
        features = text_encoder(prompts)
        text_features.append(features)
        text_labels.append(labels)


    text_features = jt.cat(text_features)
    text_labels = jt.cat(text_labels)

    # print(text_features.shape)
    # print(text_labels)

    return text_features, text_labels

# 训练和验证
def train(logit_head,
          preprocess, model_clip,
          train_loader, val_loader,
          optimizer, scheduler, criterion,
          iters, eval_freq, device="cuda"):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(iters, batch_time, losses, top1)

    if train_loader is not None:
        train_loader_iter = iter(train_loader)
    else:
        train_loader_iter = None

    result_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "text_encoder": None,
        "logit_head": None,
    }

    end = time.time()

    # 开始迭代
    for i in range(iters):
        logit_head.train()

        # 数据处理
        if train_loader_iter is not None:
            try:
                image, image_label = next(train_loader_iter)
            except StopIteration:
                image_loader_iter = iter(train_loader)
                image, image_label = next(image_loader_iter)

            # image处理 and 生产textfeature
            # 将 NumPy 数组转换为 PIL 图像并进行预处理
            images = [Image.fromarray(img_np) for img_np in image.numpy()]

            # 预处理图像得到image_feature
            images_preprocessed = jt.cat([preprocess(image).unsqueeze(0) for image in images], dim=0)
            images_preprocessed = jt.misc.to(images_preprocessed, device)
            images_clip_feature =  model_clip.encode_image(images_preprocessed)
            images_clip_feature /= images_clip_feature.norm(dim=-1, keepdim=True)


            # 得到text_feature
            text_feature, text_label = extract_text_feature(image_label.tolist(), model_clip.encode_text)

            cross_features = jt.cat([images_clip_feature, text_feature])
            cross_label = jt.cat([image_label, text_label])
            cross_label = jt.misc.to(cross_label, device)
        else:
            raise ValueError("train_loader_iter is  None")


        # 训练迭代
        logit = logit_head(cross_features)
        loss = criterion(logit, cross_label)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        scheduler.step()

        # 数据记录
        losses.update(loss.item(), logit.size(0))
        acc = (logit.argmax(1)[0].view(-1) == cross_label.float().view(-1)).float().mean() * 100
        top1.update(acc, n=logit.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # 以一定频率进行验证
        if i % eval_freq == 0:
            # 打印train的信息，然后进行val
            # progress.pr2int(i)
            val_acc = validate(logit_head, preprocess,  model_clip.encode_image, val_loader, device=device)
            print(f'iter:{i}, val_acc:{val_acc}')
            if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                result_dict["iter"] = i
                result_dict["val_acc"] = val_acc
                result_dict["image_encoder"] = deepcopy(model_clip.visual.state_dict())
                result_dict["text_encoder"] = deepcopy(model_clip.transformer.state_dict())
                result_dict["logit_head"] = deepcopy(logit_head.state_dict())

    # load best model
    model_clip.visual.load_state_dict(result_dict["image_encoder"])
    model_clip.transformer.load_state_dict(result_dict["text_encoder"])
    logit_head.load_state_dict(result_dict["logit_head"])
    val_acc = validate(logit_head, preprocess, model_clip.encode_image, val_loader, device=device)
    print(f"Best val acc: {result_dict['val_acc']:.4f} at iter {result_dict['iter']}")
    return result_dict, val_acc


def validate(logit_head, preprocess, image_encoder, val_loader, device="cuda"):

    # 记录acc
    top1 = AverageMeter('Acc@1', ':6.2f')

    with jt.no_grad():
        # for i, (image, image_label) in tqdm(enumerate(val_loader),  total=len(val_loader) // val_loader.batch_size):
        for i, (image, image_label) in enumerate(val_loader):
            # 将 NumPy 数组转换为 PIL 图像并进行预处理
            images = [Image.fromarray(img_np) for img_np in image.numpy()]

            # 预处理图像得到image_feature
            images_preprocessed = jt.cat([preprocess(image).unsqueeze(0) for image in images], dim=0)
            images_preprocessed = jt.misc.to(images_preprocessed, device)
            images_clip_feature = image_encoder(images_preprocessed)
            images_clip_feature /= images_clip_feature.norm(dim=-1, keepdim=True)

            image_label = jt.misc.to(image_label, device)

            # compute output
            logit = logit_head(images_clip_feature)

            # measure accuracy and record loss
            acc = (logit.argmax(1)[0].view(-1) == image_label.float().view(-1)).float().mean() * 100

            top1.update(acc, logit.size(0))

        # print(' * Acc@1 {top1.avg:.3f}'
        #       .format(top1=top1))
    return round(top1.avg.item(), 2)


# 数据集
class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        super().__init__()
        self.img_path = img_path
        self.img_label = img_label

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        label = self.img_label[index]

        if self.transform is not None:
            img = self.transform(img)
        # 返回必须为图片和int，不要去变成tensor
        return img, label

    def __len__(self):
        return len(self.img_path)

# 特征初始化
def process_textfeatures():
    # 加载clip
    model_clip, preprocess = clip.load("../ViT-B-32.pkl")
    model_clip.cuda()


    classes = open('../Dataset/classes_b.txt').read().splitlines()
    # remove the prefix Animal, Thu-dog, Caltech-101, Food-101

    new_classes1 = []
    new_classes2 = []
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            c1 = 'a photo of ' + c.lower().replace('_', ' ') + ', a type of animal'
            c2 = 'a photo animal of ' + c.lower().replace('_', ' ')
        if c.startswith('Thu-dog'):
            c = c[8:]
            c1 = 'a photo of ' + c.lower().replace('_', ' ') + ', a type of dog'
            c2 = 'a photo dog of ' + c.lower().replace('_', ' ')
        if c.startswith('Caltech-101'):
            c = c[12:]
            c1 = 'a photo of ' + c.lower().replace('_', ' ')
            c2 = 'a photo of ' + c.lower().replace('_', ' ')
        if c.startswith('Food-101'):
            c = c[9:]
            c1 = 'a photo of ' + c.lower().replace('_', ' ') + ', a type of food'
            c2 = 'a photo food of ' + c.lower().replace('_', ' ')
        if c.startswith('Stanford-Cars'):
            c = c[14:]
            c1 = 'a photo of ' + c.lower().replace('_', ' ') + ', a type of car'
            c2 = 'a photo car of ' + c.lower().replace('_', ' ')
        new_classes1.append(c1)
        new_classes2.append(c2)

    text1 = clip.tokenize(new_classes1)
    text2 = clip.tokenize(new_classes2)


    text_features1 = model_clip.encode_text(text1)
    text_features1 /= text_features1.norm(dim=-1, keepdim=True)

    text_features2 = model_clip.encode_text(text2)
    text_features2 /= text_features2.norm(dim=-1, keepdim=True)

    text_features = (text_features1 + text_features2) / 2.
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print(text_features.shape)
    return text_features

# 模型定义
def make_classifier_head(classifier_head,
                         clip_encoder,
                         num_classes,
                         classifier_init=True,
                         bias=False):

    if clip_encoder == 'ViT-B/32':
        in_features = 512

    linear_head = nn.Linear(in_features, num_classes, bias=bias)

    if classifier_init == True:
        # 初始化权重
        linear_head.weight.data = process_textfeatures()

    if classifier_head == 'linear':
        head = linear_head
    return head


class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale

        # Not learnable for simplicity
        # self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        self.logit_scale = jt.nn.Parameter(jt.ones([]) * logit_scale)

    def execute(self, x):
        x = x.normalize()
        # x = F.normalize(x, dim=1)
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x
    
# 模拟退火Scheduler
class WarmupScheduler(object):
    def __init__(self, optimizer, warmup_iters, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self.base_lr_pg = [pg.get("lr") for pg in optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            return self.base_lr * (self.last_epoch + 1) / self.warmup_iters
        else:
            print('超出Warmup迭代次数')

    def step(self):
        self.last_epoch += 1
        self.update_lr()

    def update_lr(self):
        self.optimizer.lr = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group.get("lr") != None:
                param_group["lr"] = self.get_lr(self.base_lr_pg[i], param_group["lr"])

class CombinedScheduler(object):
    def __init__(self, optimizer, warmup_iters, T_max):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.T_max = T_max
        self.last_epoch = -1
        self.warmup_scheduler = WarmupScheduler(optimizer, warmup_iters)
        self.cosine_scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max - warmup_iters)

    def step(self):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return self.warmup_scheduler.get_lr()
        else:
            return self.cosine_scheduler.get_lr()
        

# 基本参数设置
hyperparams = HYPER_DICT['linear']

def get_hyperparams_str(optim,
                        lr,
                        wd,
                        batch_size,
                        iters, clip_ft):
    hyperparams_str = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}_{clip_ft}"
    return hyperparams_str

def get_experiment_count(hyperparams):
    count = 1
    count *= len(hyperparams['lr'])
    count *= len(hyperparams['weight_decay'])
    count *= len(hyperparams['batch_size'])
    count *= len(hyperparams['max_iter'])
    return count

# 主函数
def main(args):
    experiment_count = get_experiment_count(hyperparams)
    cur_count = 0

    # header
    classifier_head = 'linear'
    clip_encoder = 'ViT-B/32'
    num_classes = 403

    # 预热迭代次数
    warmup_iters = 50
     # Evaluate on val set per 100 iterations (for early stopping)
    EVAL_FREQ = 100
    # Config for Adam and AdamW
    ADAM_BETAS = (0.9, 0.999)

    # default
    save_dir = '../models_pkl/'
    if args.ft_clip == "all_ft":
        save_dir = '../models_pkl/all_ft'
    elif args.ft_clip == "fronze_text":
        save_dir = '../models_pkl/frozen_text'

    # device
    device="cuda"
    # trans
    SIZE = (224, 224)

    lr = args.lr
    wd = args.wd
    batch_size = args.batch_size
    iters = args.iters

    cur_count += 1

    hyperparams_str = get_hyperparams_str(
            hyperparams['optim'], lr, wd, batch_size, iters, args.ft_clip)
    print(f"Starting: {hyperparams_str} {cur_count}/{experiment_count}")

    # 创建保存文件夹
    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # 加载clip
    model_clip, preprocess = clip.load("../ViT-B-32.pkl")

    # model创建
    head = make_classifier_head(classifier_head, clip_encoder, num_classes)
    logit_head = LogitHead(head)


    # optimizer
    params_groups = [
            {'params': logit_head.parameters()},
            {'params': model_clip.visual.parameters()},
        ]
    
    if args.ft_clip == "all_ft":
        params_groups.append({'params': model_clip.transformer.parameters()})
        
    optimizer = jt.optim.AdamW(params_groups,  lr=lr, betas=ADAM_BETAS, weight_decay=wd)
    scheduler = CombinedScheduler(optimizer, warmup_iters=warmup_iters, T_max=iters)
    criterion = nn.CrossEntropyLoss().cuda()

    # train_loader and val_loader
    train_trans = transform.Compose([
                    transform.Resize(size=max(SIZE), mode=Image.BICUBIC),
                    transform.CenterCrop(SIZE),
                    transform.RandomHorizontalFlip(),
                    # transform.ToTensor(),
                    # transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    val_trans = transform.Compose([
                    transform.Resize(size=max(SIZE), mode=Image.BICUBIC),
                    # transform.ToTensor(),
                    # transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    train_dataset = FFDIDataset(train_label['path'], train_label['target'], train_trans).set_attrs(batch_size=batch_size, shuffle=True)
    val_dataset = FFDIDataset(val_label['path'], val_label['target'], val_trans).set_attrs(batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)


    # zero-shot的验证集精确度
    val_acc = validate(logit_head, preprocess, model_clip.encode_image, val_loader, device=device)
    print(f'zero-shot的val_acc:{val_acc}')


    result_dict, val_acc = train(
            logit_head,
            preprocess, model_clip,
            train_loader, val_loader,
            optimizer, scheduler, criterion,
            iters, eval_freq=EVAL_FREQ)
    result_path = os.path.join(checkpoint_dir, f'val_result_{val_acc}_{result_dict["iter"]}.pkl')
    jt.save(result_dict, result_path)

if __name__ == "__main__":

    # 创建解析器
    parser = argparse.ArgumentParser(description="train of argparse")

    # 添加参数
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--iters", type=int, default=12800, help="Number of iterations")
    parser.add_argument("--ft_clip", type=str, default="fronze_text", help="all_ft/fronze_text")

    args = parser.parse_args()

    main(args)