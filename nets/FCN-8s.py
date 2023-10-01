import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

classess = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'potted plant',
            'sheep', 'sofa', 'train', 'tv/monitor']

colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


# 给定一个标好的图片，将像素值对应的物体类别找出来
def image2label(image, colormap):
    # 将标签转化为每个像素值为一类数据
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1] * 256 + cm[2])] = i
    # 对一张图像进行转换
    image = np.array(image, dtype="int")
    ix = (image[:, :, 0] * 256 + image[:, :, 1] * 256 + image[:, :, 2])
    image2 = cm2lbl[ix]
    return image2


# 随机裁剪图像数据
def rand_crop(data, label, high, width):
    im_width, im_high = data.size
    # 生成随机点的位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_high - high)
    right = left + width
    bottom = top + high
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    return data, label


# 单组图像的转换操作
def img_transforms(data, label, high, width, colormap):
    data, label = rand_crop(data, label, high, width)
    data_tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    data = data_tfs(data)
    label = torch.from_numpy(image2label(label, colormap))
    return data, label


# 定义需要读出的数据路径的函数
def read_image_path(root="D:\\DataSets\\PascalVOC2012\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt"):
    """"保存指定路径下的所有需要读取的图像文件路径"""
    image = np.loadtxt(root, dtype=str)
    # print("image", image)
    n = len(image)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(image):
        data[i] = "D:\\DataSets\\PascalVOC2012\\VOCdevkit\\VOC2012\\JPEGImages//%s.jpg" % (fname)
        label[i] = "D:\\DataSets\\PascalVOC2012\\VOCdevkit\\VOC2012\\SegmentationClass//%s.png" % (fname)
    return data, label


# 定义一个MyDataset继承于torch.utils.data.Dataset
class Mydataset(Data.Dataset):
    # 用于读取图像，进行相应的裁剪等
    def __init__(self, data_root, high, width, imtransform, colormap):
        # data_root 数据所对应的文件名，high,width:图像裁剪后的尺寸
        # imtransform:图像预处理操作，colormap：颜色
        self.data_root = data_root
        self.high = high
        self.width = width
        self.imtransform = imtransform
        self.colormap = colormap
        data_list, label_list = read_image_path(root=data_root)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

    def _filter(self, images):
        # 过滤掉图片大于指定图片大小的图片
        return [im for im in images if (Image.open(im).size[1] > high and Image.open(im).size[0] > width)]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.imtransform(img, label, self.high, self.width, self.colormap)
        return img, label

    def __len__(self):
        return len(self.data_list)


# 将标准化后的数据转化为0-1的区间
def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0, 1)


# 从预测的标签转化为图像的操作
def label2_image(prelabel, colormap):
    # 预测的标签转化为图像，针对一个标签图
    h, w = prelabel.shape
    prelabel = prelabel.reshape(h * w, -1)
    image = np.zeros((h * w, 3), dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)
        image[index, :] = colormap[ii]
    return image.reshape(h, w, 3)


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        # num_classes 训练数据的类别
        self.num_classes = num_classes

        # 使用预训练好的vgg19网络作为基础网络
        model_vgg19 = vgg19(pretrained=True)
        # 不使用vgg19网络中的后面的adaptiveavgpool2d和linear层
        self.base_model = model_vgg19.features
        # 定义需要的额几个层操作
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.ConvTranspose2d(32, num_classes, kernel_size=1)

        # vgg19中maxpool2所在的层
        self.layers = {"4": "max_pool_1", "9": "maxpool_2",
                       "18": "maxpool_3", "27": "maxpool_4",
                       "36": "maxpool_5"}

    def forward(self, x):
        output = {}
        for name, layer in self.base_model._modules.items():
            # 从第一层开始获取图像的特征
            x = layer(x)
            # 如果是layer中指定的特征，那就保存到output中‘
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output["maxpool_5"]
        x4 = output["maxpool_4"]
        x3 = output["maxpool_3"]

        # 对图像进行相应转置卷积操作，逐渐将图像放大到原来大小
        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score


def train_model(model, criterion, optimizer, traindataloader, valdataloader, num_epochs=25):
    since = time.time()
    best_models_wts = copy.deepcopy(model.state_dict())
    bestloss = 1e10
    train_loss_all = []
    train_acc_all = []
    val_acc_all = []
    val_loss_all = []
    since = time.time()
    for epoch in range(0, num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0
        train_num = 0
        val_loss = 0
        val_num = 0
        # 每个epoch包括训练和验证阶段
        model.train()
        for step, (b_x, b_y) in enumerate(traindataloader):
            optimizer.zero_grad()
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)  # 预测的标签
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            train_num += len(b_y)
        # 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        print('{} Train loss: {:.4f}'.format(epoch, train_loss_all[-1]))

        # 计算一个epoch在训练后在验证集上的损失和精度
        model.eval()
        for step, (b_x, b_y) in enumerate(valdataloader):
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)

        # 计算一个epoc在验证集上的损失和精度
        val_loss_all.append(val_loss / val_num)
        print('{} Val Loss:{:.4f}'.format(epoch, val_loss_all[-1]))
        # 保存最好的网络参数
        if val_loss_all[-1] < bestloss:
            bestloss = val_loss_all[-1]
            best_models_wts = copy.deepcopy(model.state_dict())
        # 每个epoch的花费时间
        time_use = time.time() - since
        print("Train and Val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "val_loss_all": val_loss_all})
    # 输出最好的模型
    model.load_state_dict(best_models_wts)
    return model, train_process


if __name__ == "__main__":
    high, width = 320, 480
    voc_train = Mydataset("D:\\DataSets\\PascalVOC2012\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt", high,
                          width,
                          img_transforms, colormap)
    voc_val = Mydataset("D:\\DataSets\\PascalVOC2012\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\val.txt", high,
                        width,
                        img_transforms, colormap)

    # 创建数据加载器，每个batch使用4张图像
    train_loader = Data.DataLoader(voc_train, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = Data.DataLoader(voc_val, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    # 可视化一个batch的数据
    # for step, (b_x, b_y) in enumerate(train_loader):
    #     if step > 0:
    #         break
    #     # 输出训练图像的尺寸和标签的尺寸，以及数据类型
    #     print("b_x.shape:", b_x.shape)
    #     print("b_y.shape:", b_y.shape)
    #
    #     b_x_numpy = b_x.data.numpy()
    #     b_x_numpy = b_x_numpy.transpose(0,2,3,1)
    #     b_y_numpy = b_y.data.numpy()
    #     plt.figure(figsize=(16,6))
    #
    #     for ii in range(4):
    #         plt.subplot(2,4,ii+1)
    #         plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    #         plt.axis("off")
    #         plt.subplot(2,4,ii+5)
    #         plt.imshow(label2_image(b_y_numpy[ii],colormap))
    #         plt.axis("off")
    #     plt.subplots_adjust(wspace=0.1,hspace=0.1)
    #     plt.show()

    fcn8s = FCN8s(21).to(device)
    summary(fcn8s, input_size=(3, high, width))
    LR = 0.0003
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(fcn8s.parameters(), lr=LR, weight_decay=1e-4)
    # d对模型进行训练，对所有的数据训练epoch轮
    fcn8s, train_process = train_model(
        fcn8s, criterion, optimizer, train_loader,
        val_loader, num_epochs=1
    )
    torch.save(fcn8s, "D:\\Lee's Net\\fcn8s.pkl")
    plt.figure(figsize=(10, 6))
    plt.plot(train_process.epoch, train_process.train_loss_all,
             "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all,
             "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()  # 训练及验证到此结束
    # 从验证集中获取一个batch的数据
    for step, (b_x, b_y) in enumerate(val_loader):
        if step > 0:
            break
        fcn8s.eval()
        b_x = b_x.float().to(device)
        b_y = b_y.long().to(device)
        out = fcn8s(b_x)
        out = F.log_softmax(out, dim=1)
        pre_lab = torch.argmax(out, 1)
        # 可视化一个batch的图像，检查数据预处理是否正确
        b_x_numpy = b_x.cpu().data.numpy()
        b_x_numpy = b_x_numpy.transpose(0, 2, 3, 1)
        b_y_numpy = b_y.cpu().data.numpy()
        pre_lab_numpy = pre_lab.cpu().data.numpy()
        plt.figure(figsize=(16, 9))
        for ii in range(4):
            plt.subplot(3, 4, ii + 1)
            plt.imshow(inv_normalize_image(b_x_numpy[ii]))
            plt.axis("off")
            plt.subplot(3, 4, ii + 5)
            plt.imshow(label2_image(b_y_numpy[ii], colormap))
            plt.axis("off")
            plt.subplot(3, 4, ii + 9)
            plt.imshow(label2_image(pre_lab_numpy[ii], colormap))
            plt.axis("off")
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()