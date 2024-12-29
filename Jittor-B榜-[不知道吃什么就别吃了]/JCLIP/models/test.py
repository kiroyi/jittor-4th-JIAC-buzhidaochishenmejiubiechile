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

test_label = pd.read_csv('../Dataset/test_b.txt')
val_label = pd.read_csv('../Dataset/valid.txt')
test_label['path'] = '../Dataset/TestSetB/' + test_label['img_name']
val_label['path'] = '../Dataset/' + val_label['img_name']


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
    
def predict(logit_head, preprocess, image_encoder, test_label, device="cuda"):

    test_num = len(test_label)

    with jt.no_grad():
        test_pred = []
        for i in tqdm(range(test_num), total=test_num):

            img_path = test_label['path'][i]

            image = Image.open(img_path)
            image_preprocessed  = preprocess(image).unsqueeze(0)

            # 预处理图像得到image_feature
            images_clip_feature = image_encoder(image_preprocessed)
            images_clip_feature /= images_clip_feature.norm(dim=-1, keepdim=True)

            # compute output
            logit = logit_head(images_clip_feature)

            logit = nn.softmax(logit, dim=1)
            logit = logit.numpy()

            test_pred.append(logit)
        test_pred = np.vstack(test_pred)
    return test_pred

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
    return round(top1.avg.item(), 3)

val_trans = transform.Compose([
                transform.Resize(size=224, mode=Image.BICUBIC),
                # transform.ToTensor(),
                # transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

val_dataset = FFDIDataset(val_label['path'], val_label['target'], val_trans).set_attrs(batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset)


# header
classifier_head = 'linear'
clip_encoder = 'ViT-B/32'
num_classes = 403

model_clip, preprocess = clip.load("../ViT-B-32.pkl")
model_path = '../models_pkl/frozen/optim_adamw-lr_0.0001-wd_0.0001-bs_4-iters_12800_fronze_text/val_result_73.98_12200.pkl'
result_dict = jt.load(model_path)

# model创建
head = make_classifier_head(classifier_head, clip_encoder, num_classes)
logit_head = LogitHead(head)

val_acc = validate(logit_head, preprocess,  model_clip.encode_image, val_loader)
print(f'zero-shot的val_acc：{val_acc}')

model_clip.visual.load_state_dict(result_dict['image_encoder'])
model_clip.transformer.load_state_dict(result_dict['text_encoder'])
logit_head.load_state_dict(result_dict["logit_head"])


val_acc = validate(logit_head, preprocess,  model_clip.encode_image, val_loader)
print(f'模型{os.path.basename(model_path)}的val_acc：{val_acc}')

test_pred = predict(logit_head, preprocess, model_clip.encode_image, test_label)
print(test_pred.shape)

top_k = 5
top_k_indices = np.argsort(test_pred, axis=1)[:, -top_k:]
top_k_indices = np.flip(top_k_indices, axis=1)

test_label[['Top1', 'Top2', 'Top3', 'Top4', 'Top5']] = top_k_indices


save_path = model_path.replace("models_pkl", "model_output")
save_path = save_path.replace(".pkl", ".txt")
print(save_path)
save_dir = os.path.dirname(save_path)
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

columns_to_save = ['img_name', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5']
df_selected = test_label[columns_to_save]
df_selected.to_csv(save_path, sep=' ', index=False, header=False)