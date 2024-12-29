import jclip as clip
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
from PIL import Image
import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
from jittor import transform, utils
from copy import deepcopy
import os
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

# load data
class TripleDataset(Dataset):
    def __init__(self, img_path, img_label, img_description, transform=None):
        super(TripleDataset, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.img_description = img_description 

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):

        img = Image.open(self.img_path[index]).convert('RGB')
        label = self.img_label[index]
        img_description = self.img_description[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_description

    def __len__(self):
        return len(self.img_path)
    
    
train_label = pd.read_csv('/mnt/workspace/JCLIP/Dataset/train_top2.txt')
val_label = pd.read_csv('/mnt/workspace/JCLIP/Dataset/valid_b.txt')
test_label = pd.read_csv('/mnt/workspace/JCLIP/Dataset/test.txt')
descri_label = pd.read_csv('/mnt/workspace/JCLIP/Dataset/descri_class_google.txt')

train_label['path'] = '/mnt/workspace/JCLIP/Dataset/' + train_label['img_name']
val_label['path'] = '/mnt/workspace/JCLIP/Dataset/' + val_label['img_name']
test_label['path'] = '/mnt/workspace/JCLIP/Dataset/TestSetB/' + test_label['img_name']
train_label['description'] = None
val_label['description'] = None
test_label['description'] = None


def append_description(dataframe):
    for index, row in dataframe.iterrows():
        label = row['target']
        descri = descri_label[descri_label['target'] == label]['class_descri'].values[0]
        dataframe.at[index, 'description'] = descri
    return dataframe


train_label = append_description(train_label)
val_label = append_description(val_label)


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
    
    
def validate(model_clip, preprocess, val_loader, device="cuda"):

    top1 = AverageMeter('Acc@1', ':6.2f')

    with jt.no_grad():
        end = time.time()
        text_features = process_textfeatures(model_clip)
        for i, (image, image_label, description) in enumerate(val_loader):
            # 将 NumPy 数组转换为 PIL 图像并进行预处理
            images = [Image.fromarray(img_np) for img_np in image.numpy()]

            # 预处理图像得到image_feature
            images_preprocessed = jt.cat([preprocess(image).unsqueeze(0) for image in images], dim=0)
            images_preprocessed = jt.misc.to(images_preprocessed, device)

            images_clip_features = model_clip.encode_image(images_preprocessed)
            # [batch_size, 512]
            images_clip_features /= images_clip_features.norm(dim=-1, keepdim=True)



            target = jt.misc.to(image_label, 'cuda')

            # compute output
            text_probs = (images_clip_features @ text_features.transpose(0, 1))

            # measure accuracy and record loss
            acc = (text_probs.argmax(1)[0].view(-1) == target.float().view(-1)).float().mean() * 100
            top1.update(acc, n=text_probs.size(0))

        return round(top1.avg.item(), 2)


def predict(test_label, model, preprocess, text_features, tta=10):


    test_num = len(test_label)

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with jt.no_grad():
            end = time.time()
            for i in tqdm(range(test_num), total=test_num):

                img_path = test_label['path'][i]
                image = Image.open(img_path)
                image = preprocess(image).unsqueeze(0)

                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # compute output
                output = (100.0 *
                      image_features @ text_features.transpose(0, 1)).softmax(
                          dim=-1)

                output = nn.softmax(output, dim=1)
                output = output.numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


def train(preprocess, model_clip,
          train_loader, val_loader,
          optimizer, scheduler, criterion_img, criterion_txt,
         iters, eval_freq=100, device="cuda"):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_text = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(iters, batch_time, losses, top1)


    # switch to train mode
    model_clip.train()

    if train_loader is not None:
        train_loader_iter = iter(train_loader)
    else:
        train_loader_iter = None

    result_dict = {
        "iter": None,
        "val_acc": None,
        "model_clip": None,
    }

    end = time.time()

    for i in range(iters):
        # 数据处理
        if train_loader_iter is not None:
            try:
                image, image_label, description = next(train_loader_iter)
            except StopIteration:
                image_loader_iter = iter(train_loader)
                image, image_label, description = next(image_loader_iter)

            # image处理 and 生产textfeature
            # 将 NumPy 数组转换为 PIL 图像并进行预处理
            images = [Image.fromarray(img_np) for img_np in image.numpy()]

            # 预处理图像
            images_preprocessed = jt.cat([preprocess(image).unsqueeze(0) for image in images], dim=0)
            images_preprocessed = jt.misc.to(images_preprocessed, device)

            # 描述
            description = clip.tokenize(description)
            description = jt.misc.to(description, 'cuda')
        else:
            raise ValueError("train_loader_iter is  None")

        # 训练迭代
        logits_per_image, logits_per_text = model_clip(images_preprocessed, description)
        ground_truth = jt.arange(len(images_preprocessed),dtype=jt.float32)
        ground_truth = jt.misc.to(ground_truth, device)

        loss = (criterion_img(logits_per_image, ground_truth) + criterion_txt(logits_per_text, ground_truth)) / 2
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        scheduler.step()

        # 数据记录
        losses.update(loss.item(), logits_per_image.size(0))

        # measure accuracy and record loss
        acc = (logits_per_image.argmax(1)[0].view(-1) == ground_truth.float().view(-1)).float().mean() * 100
        acc_text = (logits_per_text.argmax(1)[0].view(-1) == ground_truth.float().view(-1)).float().mean() * 100
        top1.update(acc, n=logits_per_image.size(0))
        top1_text.update(acc_text, n=logits_per_image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # 以一定频率进行验证
        if i % eval_freq == 0:
            # 打印train的信息，然后进行val
            # progress.pr2int(i)
            val_acc = validate(model_clip, preprocess, val_loader, device=device)
            print(f'iter:{i}, val_acc:{val_acc}')
            if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                result_dict["iter"] = i
                result_dict["val_acc"] = val_acc
                result_dict["image_encoder"] = deepcopy(model_clip.visual.state_dict())
                result_dict["text_encoder"] = deepcopy(model_clip.transformer.state_dict())

    # load best model
    model_clip.visual.load_state_dict(result_dict["image_encoder"])
    model_clip.transformer.load_state_dict(result_dict["text_encoder"])
    val_acc = validate(model_clip, preprocess, val_loader, device=device)
    print(f"Best val acc: {result_dict['val_acc']:.4f} at iter {result_dict['iter']}")
    return result_dict, val_acc

# 特征初始化
def process_textfeatures(model_clip):

    new_classes = []
    for value in descri_label['class_descri']:
        new_classes.append(value)

    # print(new_classes)

    text1 = clip.tokenize(new_classes)

    text_features = model_clip.encode_text(text1)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

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
        
def main(args):
    
    # parameter
    warmup_iters = 50
    EVAL_FREQ = 100
    ADAM_BETAS = (0.9, 0.999)
    batch_size = args.batch_size
    iters = args.iters
    lr = args.lr
    wd = args.wd
    save_dir = '/mnt/workspace/JCLIP/models_pkl/all_ft'

    # device
    device="cuda"

    # 加载clip
    model_clip, preprocess = clip.load("/mnt/workspace/JCLIP/ViT-B-32.pkl")
    model_clip.cuda()
    print('模型创建完毕')
    
    # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = jt.optim.AdamW(model_clip.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=wd)
    scheduler = CombinedScheduler(optimizer, warmup_iters=50, T_max=128000)
    # 创建损失函数
    criterion_img = nn.CrossEntropyLoss().cuda()
    criterion_txt = nn.CrossEntropyLoss().cuda()


    train_trans = transform.Compose([
                    transform.Resize((224, 224)),
                    transform.CenterCrop(224),
                    # transform.ColorJitter(0.5),
                    transform.RandomHorizontalFlip(),
                    # transform.RandomVerticalFlip(),
                    # transform.ToTensor(),
                    # transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    val_trans = transform.Compose([
                    transform.Resize((224, 224), mode=Image.BICUBIC),
                    # transform.ToTensor(),
                    # transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    train_dataset = TripleDataset(train_label['path'], train_label['target'], train_label['description'], train_trans).set_attrs(batch_size=batch_size, shuffle=True)
    val_dataset = TripleDataset(val_label['path'], val_label['target'], val_label['description'], val_trans).set_attrs(batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)

    # zero-shot的验证集精确度
    val_acc = validate(model_clip, preprocess, val_loader, device=device)
    print(f'zero-shot的val_acc:{val_acc}')

    result_dict, val_acc = train(
            preprocess, model_clip,
            train_loader, val_loader,
            optimizer, scheduler, criterion_img, criterion_txt,
            iters, eval_freq=EVAL_FREQ)

    result_path = os.path.join(save_dir, f'val_result_{val_acc}_{result_dict["iter"]}.pkl')
    jt.save(result_dict, result_path)
    
if __name__ == "__main__":

    # 创建解析器
    parser = argparse.ArgumentParser(description="train of argparse")

    # 添加参数
    parser.add_argument("--lr", type=float, default=1e-8, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--iters", type=int, default=12800, help="Number of iterations")

    args = parser.parse_args()

    main(args)