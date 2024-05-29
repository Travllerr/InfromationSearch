from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torchvision
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import argparse
from AlexNet import alex_net
from VGG import vgg


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))  # 将设备信息输出

    # 要求训练集直接返回为tensor而不是图片
    trans = [
    transforms.ToTensor(),
    transforms.Resize(224)
             ]
    trans = transforms.Compose(trans)
    # 图片直接全部读入内存中
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans
        # , download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans
        # , download=True
    )
    train_num = len(mnist_train)

    # 根据超参数设定，定义批大小
    batch_size = args.batch_size
    # 选择os.cpu_count()、batch_size、8中最小值作为num_workers
    # os.cpu_count()：Python中的方法用于获取系统中的CPU数量。如果系统中的CPU数量不确定，则此方法返回None。
    # 如果想进一步了解num_workers的知识，可以上网查阅
    nw = min([
        os.cpu_count(),
        batch_size if batch_size > 1 else 0,
        4]
    )
    print("Using {} dataloader workers every process".format(nw))  # 打印使用几个进程

    # 设置训练集dataloader。之前的train_data只是定义了整体的数据有哪些，dataloader在此基础上定义了这些数据要如何
    # 按照批大小划分给网络进行训练
    train_dataloader = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )

    # 验证集长度
    val_num = len(mnist_test)
    # 设置验证集dataloader
    val_dataloader = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw
    )

    print("Using {} for train,using {} for val".format(train_num, val_num))


    # net = alex_net()

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)


    def init_weights(m):
        """使用nn自带的xavier初始化"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)  # 将模型放入设备（cpu或者GPU）中

    learning_rate = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # 损失函数，使用交叉熵损失
    loss_function = nn.CrossEntropyLoss()

    epochs = args.epoch  # 定义训练的二篇epoch数量
    best_acc = 0.0  # 用于记录训练过程中所出现的最佳准确率，进行比较后确定是否保存模型的最优权重
    save_path = "weight/model_best.pth"  # 如果判断某个epoch的网络权重使得测试效果最佳，那么就保存该网络模型到这个路径上
    train_step = len(train_dataloader)  # 相当于一共有多少个batch
    loss_r, acc_r, acc_t = [], [], [] # 记录训练时出现的损失以及分类准确度

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        acc_train = 0.0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()  # 先进行梯度清零
            pre = net(images.to(device))  # 对类别进行预测
            predict_pre = torch.max(pre, dim=1)[1]
            acc_train += torch.eq(predict_pre, labels.to(device)).sum().item()
            loss = loss_function(pre, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # 这里就不对loss进行归一化处理了，直接算下来就是求和，确实无所谓
        loss_r.append(running_loss)
        train_accurate = acc_train / train_num
        acc_t.append(train_accurate)

        net.eval()
        acc = 0.0  # 预测正确个数
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_d in val_bar:
                val_image, val_label = val_d
                output = net(val_image.to(device))
                predict_y = torch.max(output, dim=1)[1]
                acc += torch.eq(predict_y, val_label.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        val_accurate = acc / val_num
        acc_r.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f train_accuracy: %.3f'
              % (epoch + 1, running_loss / train_step, val_accurate, train_accurate))

        if val_accurate > best_acc:  # 如果本次epoch训练的模型准确度高于之前的最高准确度，那就保存这一次模型的权重信息
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("finished training")

    with open('training_statistic.json', 'w+') as f:  # 保存训练过程的损失和准确度数据为一个json文件
        json.dump(dict(loss=loss_r, test_train=dict(accuracy_test=acc_r, accuracy_train=acc_t)), f, indent=4)


def arguments():
    """
    这个函数的作用就是专门用来定义一些必要的网络训练中要使用到的超参数和变量
    想要修改超参数及变量的值，需要在每个变量的定义里更改default字段的值
    :return:
    """
    parser = argparse.ArgumentParser(description='Arguments for training ResNet-34')

    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='网络训练的Epoch数量，即不断循环迭代训练网络的次数。建议先设置为50~100的值，然后根据测试情况逐步修改'
    )
    parser.add_argument(
        '--lr',
        type=float,
        # 即使模型很快收敛也可能需要调正学习率
        # 因为学习率会影响泛化能力，可能会到达不一样的谷底
        default=0.01,
        help='网络训练学习率。建议以10的倍数增大或减小'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        # vgg13批大小太大容易导致模型loss很高，同时训练速度也会下降
        default=64,
        help='批大小，一般设置范围在4~32之间，硬件设备性能足够时可以设置的大一些'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    main(args)
