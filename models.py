from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    输入：module_defs是一个列表，保存的是yoLov3.cfg文件的配置，每个元素是一个字典，保存的是网络的部分结构。
    module_defs是程序parse_config.py的函数parse_model_config的返回值。
    输出：hyperparams(网络超参数,一个字典), module_list(网络模型),nn.ModuleList()
    """
    # 将列表module_defs中保存网络超参数的元素去掉，并返回超参数这个字典
    hyperparams = module_defs.pop(0)#module_defs[0]存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息,把它pop出来，就不参与后面的遍历了
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()#cfg文件中的每个网络部分结构保存在一个modules中，module_list一个元素对应一个modules，即cfg文件中的一个块。
    # 开始构建网络，将网络结构保存在module_list中
    for i, module_def in enumerate(module_defs):#这里，相当于迭代原始的module_defs[1:] 而不是原始的module_defs，因为原始的[0]已经pop了
        modules = nn.Sequential()# 这里每个块用nn.sequential()创建为了一个module,一个module有多个层

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            #Add the Batch Norm Layer
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)
        #route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())#使用空的层，因为它还要执行一个非常简单的操作（加）。没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers,具体功能不在此实现，在Darknet类的forward函数中有体现"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    #以缩放到13x13为例，这里x是前面的卷积层输入特征，维度为:Bx(3x(5+80))xwxh=2x255x13x13,其中w,h为特征图的宽高
    def forward(self, x, targets=None):
        # 每个格子的anchor个数（现在是3）
        nA = self.num_anchors
        # 一个batch的图片数量
        nB = x.size(0)
        # 传入yolo层特征图宽高(这里宽高都是13，所以取一个值即可)
        nG = x.size(2)
        # 网络的步长，即输入网络图片的尺寸与最后输出的特征图的尺寸比值
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # 将2x255x13x13先view成2x3x85x13x13再permute(重排列的index)成2x3x13x13x85，
        #最后的85对应每个anchor预测出来的属性(tx,ty,tw,th,score,score_class1,score_class2...score_class80)
        # 其中tx,ty是相对于该anchor所在cell左上角的偏移坐标，代表预测出来的anchor中心坐标
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        #contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x，对应于预测坐标公式中的sigmoid(tx)，维度为2x3x13x13
        y = torch.sigmoid(prediction[..., 1])  # Center y，对应于预测坐标公式中的sigmoid(ty)
        w = prediction[..., 2]  # Width，对应于预测坐标公式中的tw
        h = prediction[..., 3]  # Height，对应于预测坐标公式中的th
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf 预测方框内含有目标的得分
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. 方框内所含目标属于每个类的概率得分

        # Calculate offsets for each grid
        # 生成所有cell的Cx坐标，一共有13x13个cell，所以x坐标有13x13个，范围从0到12。torch.arange(nG)先生成一个长度为13的行tensor
        # 再用repeat(nG,1)扩展成维度为13x13的tensor,最后用view()变成1x1x13x13的tensor,并且将类型转换成float型
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        # 得到所有经过缩小后的anchor尺寸，scaled_anchors维度为3x2,一行对应一个缩小后anchor的宽高。此时anchor的尺寸是相对于特征图
        # 特征图尺寸是原图缩放网络步长stride倍，同理anchor也缩小stride倍
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        # 得到所有缩放后anchor的宽，nA为每个cell对应的anchor的个数
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        # 得到所有缩放后anchor的高，nA为每个cell对应的anchor的个数
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors，对anchor进行平移和尺度缩放，得到预测的方框宽高
        # pred_boxes维度为2x3x13x13x4，是所有anchors预测出来的tx,ty,tw,th
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        # 对应公式 bx = sigmoid(tx) + cx，x维度为2x3x13x13,grid_x维度为1x1x13x13,相加时会根据python广播原理，扩展成2x3x13x13
        pred_boxes[..., 0] = x.data + grid_x
        # 对应公式 by = sigmoid(ty) + cy
        pred_boxes[..., 1] = y.data + grid_y
        # 对应公式 bw = pw*e^(tw),pw对应anchor_w，是anchor缩小stride倍后的宽；tw对应w.data
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        # 对应公式 bh = ph*e^(th),ph对应anchor_h，是anchor缩小stride倍后的高；th对应h.data
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()
            # 注释见util.py的build_targets函数
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            # nProposals为预测方框中，含有目标得分大于0.5的方框个数，即网络预测出来的方框。item()对只有一个元素的tensor进行操作，返回一个python数字
            nProposals = int((pred_conf > 0.5).sum().item())
            # 计算recall和precision
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            # conf_mask_true标记真正负责检测目标的anchor的位置
            conf_mask_true = mask
            # conf_mask_false标记没有负责检测目标的anchor的位置
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            # 采用均方误差计算x,y,w,h的偏移量和缩放比例的预测误差
            # ?????????? x[mask]是如何取值的? 是采用的数组索引方式进行索引，与numpy数组索引类似，但是有差异。这里实际是bool索引
            # 这里mask和x都是2x3x13x13的tensor，利用数组索引的方式提取x中的元素，被提取出来的元素就是mask中非0元素所在位置在x中对
            # 应位置的元素。所以x[mask]就将真正负责检测目标的anchor所对应的预测方框中心坐标在x方向上的偏移量的预测值提取出来。tx就
            # 是真实标签方框所对应的方框中心坐标在x方向上的偏移量。计算它们的平方误差即可。y,w,h同理
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            # 计算每个anchor预测的含有目标的损失，采用Binary Cross Entropy损失函数
            # pred_conf[conf_mask_false],长度为1005的一维tensor，提取出没有负责检测目标的anchor所预测的这个anchor含有目标的得分
            # tconf[conf_mask_false]，长度为1005的一维tensor，提取出没有负责检测目标的anchor所对应的真实目标标签，值为0
            # pred_conf[conf_mask_true]，长度为9的一维tensor，提取出真正负责检测目标的anchor所预测的这个anchor含有目标的得分
            # tconf[conf_mask_true]，长度为9的一维tensor，提取出真正负责检测目标的anchor所对应的真实目标标签，值为1
            # 这里的conf_mask_false，conf_mask_true的维度与tconf的维度都是2x3x13x13,并且conf_mask_true和mask以及tconf的维度和
            # 元素值都是相等的，等于1的元素代表这个位置对应的anchor负责检测一个目标。这里用了mask和tconf两个变量来记录，个人认为是
            # 为了用tconf[conf_mask_true]这种方式很方便的提取出真实的标签
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true])
            # 计算真正负责检测一个目标的anchor所预测的类别的得分
            # pred_cls:维度为2x3x13x13x80，是预测出来的每个anchor所含目标对应每个类别的概率，mask维度为2x3x13x13,
            # 所以pred_cls[mask]在前面的2x3x13x13的维度索引中采用的是bool值索引方式，只有mask中非0的元素在pred_cls中对应元素才会提取出来。
            # 此时提取出来的元素会自动包含最后一个没有被mask给定的维度，所以pred_cls[mask]维度为9x80，是一个二维tensor.
            # 代表真正负责目标检测的ahchor所含目标对应每个类别的概率
            # tcls:维度为2x3x13x13x80,mask维度为2x3x13x13,tcls[mask]维度为9x80，表示真实目标有9个，每个目标可能的类别有80个，
            # 只有一个元素为1，所以这80个元素中只有一个为1，用argmax得到了9x80 tensor中每行元素中最大的值对应的序号，
            # 也即类别所在序号，得到长度为9的一维tensor。
            # 参考官方文档中的公式，这里采用的是交叉熵损失函数。pred_cls[mask]对应x,torch.argmax(tcls[mask], 1)对应每个目标的class，
            # 而公式中j对应的是每个目标可能的不同的类别，这里就是x的一行中不同列的下标。最终输出的是所有目标损失值
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        # parse_model_config()函数将yolov3.cfg的各个网络层用一个列表module_defs保存，一个列表元素是一个字典，对应网络的一部分结构。
        self.module_defs = parse_model_config(config_path)
        # 得到网络的超参数hyperparams和网络模型module_list，module_list是nn.Sequential()类型
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        # module_def是一个字典，保存的是网络结构的一部分，module是torch创建的Sequential类型，对应module_def内容
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    # module[0]对应yolo层，这里将进入yolo层进行计算。*losses将losses变量变成一个list
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)
        # 是因为有三个尺度都计算了loss，所以这里除以3 
        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)  # 这里读取first 5 values头信息

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  #加载np.ndarray中的剩余权重，权重是以float32类型存储的
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:#如果 batch_normalize 的检查结果不是 True，只需要加载卷积层的偏置项
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

