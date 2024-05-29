"""
这份代码是用来计算训练好模型在测试集上的性能指标的。
计算的指标包括：①准确度（Accuracy）；②精确度（Precision）；③召回率（Recall）；④F1-Score（F1分数）
"""
import os
import argparse
import torch
import time
from PIL import Image
from torchvision import transforms
from glob import glob
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from ConfusionMatrix import ConfusionMatrix
from InformationSearch.AlexNet import alex_net
from VGG import vgg
from GoogLeNet import googlenet

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))  # 将设备信息输出

    # 要求训练集直接返回为tensor而不是图片
    trans = [transforms.ToTensor(),
             # transforms.Resize(224),
             transforms.Resize(96)
             ]
    trans = transforms.Compose(trans)
    # 图片直接全部读入内存中
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans
        # , download=True
    )

    # model = alex_net()

    # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # ratio = 4
    # small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    # model = vgg(small_conv_arch)

    model = googlenet()

    # 通过指定的权重文件路径，向网络模型上加载这个权重
    weight_path = args.weight_path
    assert os.path.exists(weight_path), f"file: '{weight_path}' dose not exist."
    model.load_state_dict(torch.load(weight_path))  # 网络模型加载权重
    model.to(device)

    model.eval()  # 将模型设定为验证模式
    batch_size = args.batch_size  # 每次预测时将多少张图片打包成一个batch

    # 验证集长度
    val_num = len(mnist_test)
    # 设置验证集dataloader
    val_dataloader = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    # 10分类问题实例化混淆矩阵，这里NUM_CLASSES = 5
    confusion = ConfusionMatrix(num_classes=10, labels=text_labels)

    with torch.no_grad():
        for val_d in val_dataloader:
            val_image, val_label = val_d
            output = model(val_image.to(device))
            predict_y = torch.max(output, dim=1)[1]
            # 可能是只有cpu的tensor才能转换成numpy
            confusion.update(predict_y.cpu().numpy(), val_label.cpu().numpy())
        confusion.plot()

def arguments():
    """
    用于定义各类要用到的超参数和变量。如果要对参数进行改变，只需要在对应的参数内修改default值即可
    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight_path',
        type=str,
        default='weight/model_best.pth',
        help='保存的最优模型权重'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='批大小，建议设定范围为4~32（2的倍数)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    main(args)
