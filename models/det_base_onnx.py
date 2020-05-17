from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import time
# from pypcd import pypcd
import numpy as np
from io import StringIO  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from configs.config import cfg
from datasets.dataset_info import KITTICategory

from models.model_util import get_box3d_corners_helper
from models.model_util import huber_loss

from models.common import Conv1d, Conv2d, DeConv1d, init_params
from models.common import softmax_focal_loss_ignore, get_accuracy

from ops.query_depth_point.query_depth_point import QueryDepthPoint
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from models.box_transform import size_decode, size_encode, center_decode, center_encode, angle_decode, angle_encode


NUM_SIZE_CLUSTER = len(KITTICategory.CLASSES)
MEAN_SIZE_ARRAY = KITTICategory.MEAN_SIZE_ARRAY


# single scale PointNet module
class PointNetModule(nn.Module):
    def __init__(self, Infea, mlp, dist, nsample, use_xyz=True, use_feature=True,npoint=280):
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.npoint=npoint
        if Infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, npoint)

        if self.use_xyz:
            self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        else:
            self.conv1 = Conv2d(Infea, mlp[0], 1)

        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)

        init_params([self.conv1[0], self.conv2[0], self.conv3[0]], 'kaiming_normal')
        init_params([self.conv1[1], self.conv2[1], self.conv3[1]], 1)

    def forward(self,feat=None,valid=None,grouped_pc=None):
        batch_size = 1

        npoint = self.npoint

        k = self.nsample

        if self.use_feature:
            grouped_feature = torch.gather(
                feat, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1)
            ).view(batch_size, feat.size(1), npoint, k)

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)

        grouped_feature = self.conv1(grouped_pc)
        grouped_feature = self.conv2(grouped_feature)
        grouped_feature = self.conv3(grouped_feature)
        
        grouped_feature = grouped_feature * valid

        return grouped_feature


# multi-scale PointNet module
class PointNetFeat(nn.Module):
    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()

        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        #assert len(u) == 4
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 32, use_xyz=True, use_feature=True,npoint=280)

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 64, use_xyz=True, use_feature=True,npoint=140)

        self.pointnet3 = PointNetModule(
            input_channel - 3, [128, 128, 256], u[2], 64, use_xyz=True, use_feature=True,npoint=70)

        self.pointnet4 = PointNetModule(
            input_channel - 3, [256, 256, 512], u[3], 128, use_xyz=True, use_feature=True,npoint=35)

    def forward(self, one_hot_vec1=None,one_hot_vec2=None,one_hot_vec3=None,one_hot_vec4=None,listPC=None,listInputs=None):


        # torch.onnx.export(self.pointnet1, (pc,pc1), "./frustrumPointnet1.onnx", export_params=True, opset_version=10,do_constant_folding=False)
        #assert 1/0
        feat1 = self.pointnet1(None,listInputs[0],listInputs[1])
        print("DWD")
        print(feat1.shape)
        feat1 = feat1[:,:,:,0]


        feat2 = self.pointnet2(None,listInputs[2],listInputs[3])
        feat2 = feat2[:,:,:,0]

        feat3 = self.pointnet3(None,listInputs[4],listInputs[5])
        feat3 = feat3[:,:,:,0]

        feat4 = self.pointnet4(None,listInputs[6],listInputs[7])
        feat4 = feat4[:,:,:,0]

        if one_hot_vec1 is not None:
            feat1 = torch.cat([feat1, one_hot_vec1], 1)
            feat2 = torch.cat([feat2, one_hot_vec2], 1)
            feat3 = torch.cat([feat3, one_hot_vec3], 1)
            feat4 = torch.cat([feat4, one_hot_vec4], 1)

        return [feat1, feat2, feat3, feat4]


# FCN
class ConvFeatNet(nn.Module):
    def __init__(self, i_c=128, num_vec=3):
        super(ConvFeatNet, self).__init__()
        self.block1_conv1 = Conv1d(i_c + num_vec, 128, 3, 1, 1)

        self.block2_conv1 = Conv1d(128, 128, 3, 2, 1)
        self.block2_conv2 = Conv1d(128, 128, 3, 1, 1)
        self.block2_merge = Conv1d(128 + 128 + num_vec, 128, 1, 1)

        self.block3_conv1 = Conv1d(128, 256, 3, 2, 1)
        self.block3_conv2 = Conv1d(256, 256, 3, 1, 1)
        self.block3_merge = Conv1d(256 + 256 + num_vec, 256, 1, 1)

        self.block4_conv1 = Conv1d(256, 512, 3, 2, 1)
        self.block4_conv2 = Conv1d(512, 512, 3, 1, 1)
        self.block4_merge = Conv1d(512 + 512 + num_vec, 512, 1, 1)

        self.block2_deconv = DeConv1d(128, 256, 1, 1, 0)
        self.block3_deconv = DeConv1d(256, 256, 2, 2, 0)
        self.block4_deconv = DeConv1d(512, 256, 4, 4, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):
        x = self.block1_conv1(x1)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = torch.cat([x, x2], 1)
 
        x = self.block2_merge(x)
        xx1 = x

        x = self.block3_conv1(x)

        x = self.block3_conv2(x)

        x = torch.cat([x, x3], 1)
        x = self.block3_merge(x)
        xx2 = x

        x = self.block4_conv1(x)

        x = self.block4_conv2(x)

        x = torch.cat([x, x4], 1)
        x = self.block4_merge(x)
        xx3 = x

        xx1 = self.block2_deconv(xx1)

        xx2 = self.block3_deconv(xx2)

        xx3 = self.block4_deconv(xx3)

        x = torch.cat([xx1, xx2[:, :, :xx1.shape[-1]], xx3[:, :, :xx1.shape[-1]]], 1)

        return x


# the whole pipeline
class PointNetDet(nn.Module):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__()

        self.feat_net = PointNetFeat(input_channel, 0)
        self.conv_net = ConvFeatNet()

        self.num_classes = num_classes

        num_bins = cfg.DATA.NUM_HEADING_BIN
        self.num_bins = num_bins

        output_size = 3 + num_bins * self.num_classes + NUM_SIZE_CLUSTER * 4

        self.reg_out = nn.Conv1d(768, output_size, 1)
        self.cls_out = nn.Conv1d(768, 2, 1)
        self.relu = nn.ReLU(True)

        nn.init.kaiming_uniform_(self.cls_out.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.reg_out.weight, mode='fan_in')

        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

    def _slice_output(self, output):

        batch_size = 140

        num_bins = self.num_bins

        center = output[:, 0:3].contiguous()

        heading_scores = output[:, 3:3 + num_bins].contiguous()

        heading_res_norm = output[:, 3 + num_bins:3 + num_bins * 2].contiguous()

        size_scores = output[:, 3 + num_bins * 2:3 + num_bins * 2 + NUM_SIZE_CLUSTER].contiguous()

        size_res_norm = output[:, 3 + num_bins * 2 + NUM_SIZE_CLUSTER:].contiguous()
        size_res_norm = size_res_norm.view(batch_size, NUM_SIZE_CLUSTER, 3)

        return center, heading_scores, heading_res_norm, size_scores, size_res_norm
    def setInput(self,inputvec):
        self.inputvec=inputvec
        
    def get_center_loss(self, pred_offsets, gt_offsets):

        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)

        return center_loss

    def get_heading_loss(self, heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label):

        heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)

        # b, NUM_HEADING_BIN -> b, 1
        heading_res_norm_select = torch.gather(heading_res_norm, 1, heading_class_label.view(-1, 1))

        heading_res_norm_loss = huber_loss(
            heading_res_norm_select.squeeze(1) - heading_res_norm_label, delta=1.0)

        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(self, size_scores, size_res_norm, size_class_label, size_res_label_norm):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)

        # b, NUM_SIZE_CLUSTER, 3 -> b, 1, 3
        size_res_norm_select = torch.gather(size_res_norm, 1,
                                            size_class_label.view(batch_size, 1, 1).expand(
                                                batch_size, 1, 3))

        size_norm_dist = torch.norm(
            size_res_label_norm - size_res_norm_select.squeeze(1), 2, dim=-1)

        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)

        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):

        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds

        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)

        corners_3d_pred = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

        # N, 8, 3
        corners_dist = torch.min(
            torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1).mean(-1),
            torch.norm(corners_3d_pred - corners_3d_gt_flip, 2, dim=-1).mean(-1))
        # corners_dist = torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1)
        corners_loss = huber_loss(corners_dist, delta=1.0)

        return corners_loss, corners_3d_gt

    def forward(self,inputvec):

        one_hot_vec1,one_hot_vec2,one_hot_vec3,one_hot_vec4,_,_,_,_,_,_,_,_,_ = inputvec


        cls_label = None #data_dicts.get('label')
        size_class_label = None  #data_dicts.get('size_class')
        center_label =  None #data_dicts.get('box3d_center')
        heading_label = None #data_dicts.get('box3d_heading')
        size_label = None #data_dicts.get('box3d_size')

        batch_size = 1

        feat = self.feat_net(            
            one_hot_vec1,one_hot_vec2,one_hot_vec3,one_hot_vec4,inputvec[4:5],inputvec[5:])
        feat1, feat2, feat3, feat4=feat

        x = self.conv_net(feat1, feat2, feat3, feat4)
        
        cls_scores = self.cls_out(x)#.unsqueeze(1)
        outputs = self.reg_out(x)#.unsqueeze(1)
        outputs = (cls_scores, outputs)
      
        return outputs
        




