
import numpy as np
import torch
import time

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution).cuda()
    x2 = torch.FloatTensor(1, *resolution).cuda()
    return dict(RGB=x1, depth=x2)

if __name__ == '__main__':
    #卷积加速
    #torch.backends.cudnn.benchmark = True

    # 加载网络
    from model.model_fusion import FusionNet

    channels = [16, 24, 40, 112, 160]
    model_fusion = FusionNet(channels)
    model_fusion = model_fusion.eval()

    #查看网络结构
    from torchsummary import summary
    summary(model_fusion.cuda(),input_size=[(3,256,256), (3,256,256)])

    #计算网络的计算量
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model_fusion.cuda(), (3,256,256),
                                             input_constructor=prepare_input,
                                             as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    img = torch.randn(1,3,256,256).cuda()
    depth = torch.randn(1,3,256,256).cuda()

    time_spent = []
    for idx in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            p = model_fusion(img, depth)

        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        end = time.time()

        if idx > 10:
            time_spent.append(end - start_time)

    print('Avg execution time (ms): {:.2f}'.format(np.mean(time_spent)*1000))
    #print(time_spent)


