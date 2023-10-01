import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.CBAM_ASPP import CBAMLayer


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # 输入shape为576*576*3
        low_level_features = self.features[:4](x)  # 144*144*24
        the_three_features = self.features[:7](x)  # 72*72*32
        the_four_features = self.features[:11](x)  # 36*36*64
        x = self.features[4:](low_level_features)  # 36*36*320
        return low_level_features, the_three_features, the_four_features, x

    # -----------------------------------------#


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class CBAM_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(CBAM_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=16 * rate, dilation=16 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch6_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch6_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch6_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
        nn.Conv2d(dim_out * 6, dim_out, 1, 1, padding=0, bias=True),
        nn.BatchNorm2d(dim_out, momentum=bn_mom),
        nn.ReLU(inplace=True),
        )
        self.cbam = CBAMLayer(channel=dim_out * 6)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        conv3x3_4 = self.branch5(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch6_conv(global_feature)
        global_feature = self.branch6_bn(global_feature)
        global_feature = self.branch6_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature], dim=1)
        #   加入cbam
        cbamaspp = self.cbam(feature_cat)
        result = self.conv_cat(cbamaspp)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=8):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
            # the_three_channels = 32
            # the_four_channels = 64
            
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))
        # CA
        # self.CA = CoordAtt(320, 320)
        # self.CA1 = CoordAtt(24, 24)
        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = CBAM_ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        # self.CA2 = CoordAtt(256, 256)
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

        # CFF
        self.F1 = nn.Sequential(
            nn.Conv2d(32, 192, 1, stride=1, padding=0),
            nn.BatchNorm2d(192)
        )
        self.F2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2, bias=True),  # dilation=2的膨胀卷积
            nn.BatchNorm2d(64, momentum=0.1),
        )

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, the_three_features, the_four_features, x = self.backbone(x)
        # x = self.CA(x)
        x = self.aspp(x)
        # low_level_features = self.CA1(low_level_features)
        low_level_features = self.shortcut_conv(low_level_features)

        # ---------------
        F1 = self.F1(the_three_features)  # 72*72*32-72*72*192
        # 36*36*64-72*72*64
        F2_0 = F.interpolate(the_four_features, size=(the_three_features.size(2), the_three_features.size(3)), mode='bilinear',
                             align_corners=True)
        F2_1 = self.F2_1(F2_0)  # 72*72*64-72*72*64
        FN = F.relu_(torch.cat((F1, F2_1), dim=1))  # 72*72*256
        # ----------------------------------------#
        x = F.interpolate(x, size=(the_three_features.size(2), the_three_features.size(3)), mode='bilinear',
                          align_corners=True)  # 72*72*256
        FN2 = FN + x  # 72*72*256
        F2_1 = F.interpolate(FN2, size=(low_level_features.size(2), low_level_features.size(3)),
                             mode='bilinear', align_corners=True)  # 144*144*256

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        # x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        # x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cat_conv(torch.cat((low_level_features, F2_1), dim=1))  # 144*144*304-144*144*256
        # x = self.CA2(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x




# CA
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class h_sigmoid(nn.Module):
	def __init__(self, inplace=True):
		super(h_sigmoid, self).__init__()
		self.relu = nn.ReLU6(inplace=inplace)
 
	def forward(self, x):
		return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
	def __init__(self, inplace=True):
		super(h_swish, self).__init__()
		self.sigmoid = h_sigmoid(inplace=inplace)
 
	def forward(self, x):
		return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
	def __init__(self, inp, oup, reduction=32):
		super(CoordAtt, self).__init__()
		self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
		self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
		mip = max(8, inp // reduction)
 
		self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(mip)
		self.act = h_swish()
 
		self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
		self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
 
	def forward(self, x):
		identity = x
 
		n, c, h, w = x.size()
		x_h = self.pool_h(x)
		x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
		y = torch.cat([x_h, x_w], dim=2)
		y = self.conv1(y)
		y = self.bn1(y)
		y = self.act(y)
 
		x_h, x_w = torch.split(y, [h, w], dim=2)
		x_w = x_w.permute(0, 1, 3, 2)
 
		a_h = self.conv_h(x_h).sigmoid()
		a_w = self.conv_w(x_w).sigmoid()
 
		out = identity * a_w * a_h
 
		return out
