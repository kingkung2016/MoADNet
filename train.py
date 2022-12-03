
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

#from dataset_loader import MyData
from dataset_loader_augment import MyData
#加载网络
from model.model_fusion import FusionNet
#加载损失函数
import pytorch_losses


def Hybrid_Loss(pred, target, reduction='mean'):
    #先对输出做归一化处理
    pred = torch.sigmoid(pred)

    #BCE LOSS
    bce_loss = nn.BCELoss()
    bce_out = bce_loss(pred, target)

    #IOU LOSS
    iou_loss = pytorch_losses.IOU(reduction=reduction)
    iou_out = iou_loss(pred, target)

    #SSIM LOSS
    ssim_loss = pytorch_losses.SSIM(window_size=11)
    ssim_out = ssim_loss(pred, target)


    hybrid_loss = [bce_out, iou_out, ssim_out]
    losses = bce_out + iou_out + ssim_out

    return hybrid_loss, losses


def cross_entropy2d_edge(input, target, reduction='mean'):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


#获取当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#训练一个epoch
class Trainer(object):
    def __init__(self, cuda, model_fusion, optimizer, scheduler, train_loader, epochs, save_epoch):
        self.cuda = cuda
        self.model_fusion = model_fusion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.present_epoch = 0
        self.epochs = epochs          #总的epoch
        self.save_epoch = save_epoch    #间隔save_epoch保存一次权重文件

        self.train_loss_list = []
        self.train_log = './train_log.txt'
        with open(self.train_log, 'w') as f:
            f.write(('%10s' * 8) % ('Epoch', 'losses', 'loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5', 'lr'))
        f.close()

    def train_epoch(self):
        print(('\n' + '%10s' * 8) % ('Epoch', 'losses', 'loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5', 'lr'))
        # 计算所有的loss
        losses_all, loss_1_all, loss_2_all, loss_3_all, loss_4_all, loss_5_all = 0, 0, 0, 0, 0, 0
        Hybrid_losses_1,Hybrid_losses_3,Hybrid_losses_4,Hybrid_losses_5 = [0,0,0],[0,0,0],[0,0,0],[0,0,0]

        #设置进度条
        with tqdm(total=len(self.train_loader)) as pbar:
            for batch_idx, (img, mask, depth, edge) in enumerate(self.train_loader):

                if self.cuda:
                    img, mask, depth, edge = img.cuda(), mask.cuda(), depth.cuda(), edge.cuda()
                    img, mask, depth, edge = Variable(img), Variable(mask), Variable(depth), Variable(edge)
                n, c, h, w = img.size()  # batch_size, channels, height, weight
                #梯度清零
                self.optimizer.zero_grad()

                depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)  #把深度图变成3个通道
                # depth = depth.view(n, 1, h, w)
                mask = mask.view(n, 1, h, w)
                edge = edge.view(n, 1, h, w)
                #前向传播
                F1_out, F2_out, F3_out, F4_out, F5_out = self.model_fusion(img, depth)

                #计算损失函数
                mask = mask.to(torch.float32)
                edge = edge.to(torch.float32)

                hybrid_loss_1, loss_1 = Hybrid_Loss(F1_out, mask)
                loss_2 = cross_entropy2d_edge(F2_out, edge)
                hybrid_loss_3, loss_3 = Hybrid_Loss(F3_out, mask)
                hybrid_loss_4, loss_4 = Hybrid_Loss(F4_out, mask)
                hybrid_loss_5, loss_5 = Hybrid_Loss(F5_out, mask)

                #每一个stage的损失函数求和
                Hybrid_losses_1 = [Hybrid_losses_1[i] + hybrid_loss_1[i].item() for i in range(len(hybrid_loss_1))]
                Hybrid_losses_3 = [Hybrid_losses_3[i] + hybrid_loss_3[i].item() for i in range(len(hybrid_loss_3))]
                Hybrid_losses_4 = [Hybrid_losses_4[i] + hybrid_loss_4[i].item() for i in range(len(hybrid_loss_4))]
                Hybrid_losses_5 = [Hybrid_losses_5[i] + hybrid_loss_5[i].item() for i in range(len(hybrid_loss_5))]

                #计算损失函数用于反向传播
                loss_1_all += loss_1.item()
                loss_2_all += loss_2.item()
                loss_3_all += loss_3.item()
                loss_4_all += loss_4.item()
                loss_5_all += loss_5.item()
                losses = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
                losses_all += losses.item()

                #实时更新信息
                s = ('%10s' * 1 + '%10.4g' * 7) % (self.present_epoch, losses.item(), loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item(), get_lr(self.optimizer))
                pbar.set_description(s)
                pbar.update(1)

                #反向传播
                losses.backward()
                #更新权重
                self.optimizer.step()

        # 保存模型
        if self.present_epoch % self.save_epoch == 0:
            savename_ladder = ('checkpoint/RGBD-SOD_iter_%d.pth' % (self.present_epoch))
            torch.save(self.model_fusion.state_dict(), savename_ladder)

        total_batch = len(self.train_loader)
        #输出最后的损失函数
        epoch_information = ('\n' + '%10s' * 1 + '%10.4g' * 7) % ((self.present_epoch),
        losses_all / total_batch, loss_1_all / total_batch, loss_2_all / total_batch,
        loss_3_all / total_batch, loss_4_all / total_batch, loss_5_all / total_batch,
        get_lr(self.optimizer))
        print(epoch_information)

        #输出各个损失函数的值
        print(('Stage' + '%10s' * 3) % ('bce_out', 'iou_out', 'ssim_out'))
        print(('1' + '%10.4g' * 3) % (
        Hybrid_losses_1[0] / total_batch, Hybrid_losses_1[1] / total_batch, Hybrid_losses_1[2] / total_batch))
        print(('3' + '%10.4g' * 3) % (
        Hybrid_losses_3[0] / total_batch, Hybrid_losses_3[1] / total_batch, Hybrid_losses_3[2] / total_batch))
        print(('4' + '%10.4g' * 3) % (
        Hybrid_losses_4[0] / total_batch, Hybrid_losses_4[1] / total_batch, Hybrid_losses_4[2] / total_batch))
        print(('5' + '%10.4g' * 3) % (
        Hybrid_losses_5[0] / total_batch, Hybrid_losses_5[1] / total_batch, Hybrid_losses_5[2] / total_batch))

        #写入文件
        with open(self.train_log, 'a') as f:
            f.write(epoch_information)
        f.close()

        #更新学习率
        self.scheduler.step()

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.present_epoch += 1



if __name__ == '__main__':
    """
        opt参数解析：
        pre-trained: 是否加载预训练模型
        checkpoint: 预训练模型的路径
        train-root: 训练数据集路径
        epochs: 训练总轮次
        batch-size: 批次大小
        workers: dataloader的最大worker数量
        save-epoch: 间隔save-epoch保存一次模型
        cuda: 是否使用GPU进行训练
        GPU-id: 使用单块GPU时设置的编号
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='path to pre-trained parameters')
    parser.add_argument('--train-root', type=str, default='/mnt/02AA93C51773C62F/dataset/train_2985/', help='path to the train dataset')
    #超参数设置
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--save-epoch', type=int, default=1)
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--GPU-id', type=int, default=0)
    args = parser.parse_args()

    #加载训练集
    train_loader = torch.utils.data.DataLoader(MyData(args.train_root),batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)

    # 定义网络结构
    channels = [16, 24, 40, 112, 160]
    model_fusion = FusionNet(channels)


    # 使用GPU
    if args.cuda:
        assert torch.cuda.is_available, 'ERROR: cuda can not use'
        torch.cuda.set_device(args.GPU_id)  #指定显卡
        #torch.cuda.set_device('cuda:' + str(gpu_ids))  # 可指定多卡
        #torch.backends.cudnn.benchmark = True  # GPU网络加速
        model_fusion = model_fusion.cuda()
        #model_fusion = torch.nn.DataParallel(model_fusion)  #多GPU训练

    #定义优化器
    optimizer = optim.Adam(model_fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model_fusion.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    # 等间隔调整学习率，每训练step_size个epoch，lr*gamma
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # 多间隔调整学习率，每训练至milestones中的epoch，lr*gamma
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.1)

    # 指数学习率衰减，lr*gamma**epoch
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 余弦退火学习率衰减，T_max表示半个周期，lr的初始值作为余弦函数0处的极大值逐渐开始下降，
    # 在epoch=T_max时lr降至最小值，即pi/2处，然后进入后半个周期，lr增大
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    #开始训练
    training = Trainer(
        cuda=args.cuda,
        model_fusion=model_fusion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_loader=train_loader,
        epochs=args.epochs,
        save_epoch=args.save_epoch
    )
    training.train()










