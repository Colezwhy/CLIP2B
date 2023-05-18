import copy


from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import build_head
import torch.nn as nn

from .utils import tokenize



def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    base_scales = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']
    if 'cut_mode' in proposal_cfg:
        cut_mode = proposal_cfg['cut_mode']
    else:
        cut_mode = 'symmetry'
    base_proposal_list = []
    proposals_valid_list = []
    for i in range(len(gt_points)):
        img_h, img_w, _ = img_meta[i]['img_shape']
        base = min(img_w, img_h) / 100
        base_proposals = []
        for scale in base_scales:
            scale = scale * base
            for ratio in base_ratios:
                base_proposals.append(gt_points[i].new_tensor([[scale * ratio, scale / ratio]]))

        base_proposals = torch.cat(base_proposals)
        base_proposals = base_proposals.repeat((len(gt_points[i]), 1))
        base_center = torch.repeat_interleave(gt_points[i], len(base_scales) * len(base_ratios), dim=0)

        if shake_ratio is not None:
            base_x_l = base_center[:, 0] - shake_ratio * base_proposals[:, 0]
            base_x_r = base_center[:, 0] + shake_ratio * base_proposals[:, 0]
            base_y_t = base_center[:, 1] - shake_ratio * base_proposals[:, 1]
            base_y_d = base_center[:, 1] + shake_ratio * base_proposals[:, 1]
            if cut_mode is not None:
                base_x_l = torch.clamp(base_x_l, 1, img_w - 1)
                base_x_r = torch.clamp(base_x_r, 1, img_w - 1)
                base_y_t = torch.clamp(base_y_t, 1, img_h - 1)
                base_y_d = torch.clamp(base_y_d, 1, img_h - 1)

            base_center_l = torch.stack([base_x_l, base_center[:, 1]], dim=1)
            base_center_r = torch.stack([base_x_r, base_center[:, 1]], dim=1)
            base_center_t = torch.stack([base_center[:, 0], base_y_t], dim=1)
            base_center_d = torch.stack([base_center[:, 0], base_y_d], dim=1)

            shake_mode = 0
            if shake_mode == 0:
                base_proposals = base_proposals.unsqueeze(1).repeat((1, 5, 1))
            elif shake_mode == 1:
                base_proposals_l = torch.stack([((base_center[:, 0] - base_x_l) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_r = torch.stack([((base_x_r - base_center[:, 0]) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_t = torch.stack([base_proposals[:, 0],
                                                ((base_center[:, 1] - base_y_t) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals_d = torch.stack([base_proposals[:, 0],
                                                ((base_y_d - base_center[:, 1]) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals = torch.stack(
                    [base_proposals, base_proposals_l, base_proposals_r, base_proposals_t, base_proposals_d], dim=1)

            base_center = torch.stack([base_center, base_center_l, base_center_r, base_center_t, base_center_d], dim=1)

        if cut_mode == 'symmetry':
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * base_center[..., 0])
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * (img_w - base_center[..., 0]))
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * base_center[..., 1])
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * (img_h - base_center[..., 1]))

        base_proposals = torch.cat([base_center, base_proposals], dim=-1)
        base_proposals = base_proposals.reshape(-1, 4)
        base_proposals = bbox_cxcywh_to_xyxy(base_proposals)
        proposals_valid = base_proposals.new_full(
            (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
        if cut_mode == 'clamp':
            base_proposals[..., 0:4:2] = torch.clamp(base_proposals[..., 0:4:2], 0, img_w)
            base_proposals[..., 1:4:2] = torch.clamp(base_proposals[..., 1:4:2], 0, img_h)
            proposals_valid_list.append(proposals_valid)
        if cut_mode == 'symmetry':
            proposals_valid_list.append(proposals_valid)
        elif cut_mode == 'ignore':
            img_xyxy = base_proposals.new_tensor([0, 0, img_w, img_h])
            iof_in_img = bbox_overlaps(base_proposals, img_xyxy.unsqueeze(0), mode='iof')
            proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)
        elif cut_mode is None:
            proposals_valid_list.append(proposals_valid)
        base_proposal_list.append(base_proposals)

    return base_proposal_list, proposals_valid_list


def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        pos_box = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        neg_bboxes = torch.stack([x1, y1, x2, y2], dim=1).to(gt_points[0].device)
        gt_point = gt_points[i]
        gt_min_box = torch.cat([gt_point - 10, gt_point + 10], dim=1)
        iou = bbox_overlaps(neg_bboxes, pos_box)
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])

        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
    return neg_proposal_list, neg_weight_list


def fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    # cut_mode = fine_proposal_cfg['cut_mode']
    cut_mode = None
    if isinstance(fine_proposal_cfg['base_ratios'], tuple):
        base_ratios = fine_proposal_cfg['base_ratios'][stage - 1]
        shake_ratio = fine_proposal_cfg['shake_ratio'][stage - 1]
    else:
        base_ratios = fine_proposal_cfg['base_ratios']
        shake_ratio = fine_proposal_cfg['shake_ratio']
    if gen_mode == 'fix_gen':
        proposal_list = []
        proposals_valid_list = []
        for i in range(len(img_meta)):
            pps = []
            base_boxes = pseudo_boxes[i]
            for ratio_w in base_ratios:
                for ratio_h in base_ratios:
                    base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)
                    base_boxes_[:, 2] *= ratio_w
                    base_boxes_[:, 3] *= ratio_h
                    base_boxes_ = bbox_cxcywh_to_xyxy(base_boxes_)
                    pps.append(base_boxes_.unsqueeze(1))
            pps_old = torch.cat(pps, dim=1)
            if shake_ratio is not None:
                pps_new = []

                pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))
                for ratio in shake_ratio:
                    pps = bbox_xyxy_to_cxcywh(pps_old)
                    pps_center = pps[:, :, :2]
                    pps_wh = pps[:, :, 2:4]
                    pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                    pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                    pps_y_t = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]
                    pps_y_d = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]
                    pps_center_l = torch.stack([pps_x_l, pps_center[:, :, 1]], dim=-1)
                    pps_center_r = torch.stack([pps_x_r, pps_center[:, :, 1]], dim=-1)
                    pps_center_t = torch.stack([pps_center[:, :, 0], pps_y_t], dim=-1)
                    pps_center_d = torch.stack([pps_center[:, :, 0], pps_y_d], dim=-1)
                    pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
                    pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                    pps = torch.cat([pps_center, pps_wh], dim=-1)
                    pps = pps.reshape(pps.shape[0], -1, 4)
                    pps = bbox_cxcywh_to_xyxy(pps)
                    pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 4))
                pps_new = torch.cat(pps_new, dim=2)
            else:
                pps_new = pps_old
            h, w, _ = img_meta[i]['img_shape']
            if cut_mode is 'clamp':
                pps_new[..., 0:4:2] = torch.clamp(pps_new[..., 0:4:2], 0, w)
                pps_new[..., 1:4:2] = torch.clamp(pps_new[..., 1:4:2], 0, h)
                proposals_valid_list.append(pps_new.new_full(
                    (*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
            else:
                img_xyxy = pps_new.new_tensor([0, 0, w, h])
                iof_in_img = bbox_overlaps(pps_new.reshape(-1, 4), img_xyxy.unsqueeze(0), mode='iof')
                proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)

            proposal_list.append(pps_new.reshape(-1, 4))

    return proposal_list, proposals_valid_list


#######################Edited by colez.
@DETECTORS.register_module()
class CLIP2B(TwoStageDetector):
    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 context_length,
                 class_names,
                 roi_head,
                 tau=0.07,
                 token_embed_dim=512,
                 text_dim=1024,
                 gen_range = None,
                 seg_loss=False,
                 point_ambience = False,
                 clip_head=True,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained

            self.backbone = build_backbone(backbone)
            self.text_encoder = build_backbone(text_encoder)
            self.context_decoder = build_backbone(context_decoder)
            self.context_length = context_length

            self.tau = tau
            if neck is not None:
                self.neck = build_neck(neck)
            if bbox_head is not None:
                self.with_bbox_head = True
                self.bbox_head = build_head(bbox_head)
                bbox_head.update(train_cfg=train_cfg)
                bbox_head.update(test_cfg=test_cfg)
            self.roi_head = build_head(roi_head)
            self.num_stages = roi_head.num_stages
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg

            self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
            self.use_seg_loss = seg_loss
            self.class_names = class_names
            self.clip_head = clip_head
            self.use_pt_ambience = point_ambience
            self.gen_range = gen_range

            if self.use_pt_ambience and self.gen_range is None:
                raise ValueError('A gen_range is needed when point ambience is used.')
            # learnable textual contexts
            context_length = self.text_encoder.context_length - self.context_length
            # cxt_length in text_encoder is the full length of the input of text encoder
            self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
            self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
            # creating an optimizable parameter for backprop


    def extract_feat(self, img, use_seg_loss=False, dummy=False, use_pt_ambience=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)

        # feature map from CLIPRes+AttentionPool and the final output being
        text_features = self.compute_text_features(x, dummy=dummy)
        # from prompt to text encoders and produce feat(K * C)
        score_maps = self.compute_score_maps(x, text_features)
        x = list(x[:-1])


        # for i in range(len(x)):
        #     print(x[i].shape)



        # the output of the last layer is thrown away

        # print(x[3].shape, score_maps[3].shape)

        x[3] = torch.cat([x[3], score_maps[3]], dim=1)
        # changed into x[2], originally 3
        if self.with_neck:
            x = self.neck(x)
            # for example: FPN
        if use_seg_loss or use_pt_ambience:
            return x, score_maps[0]
        # the score_map[0] can be viewed as a segmentation map for dense prediction,try if the score_maps[3] have some usages
        else:
            return x

    def compute_score_maps(self, x, text_features):
        # B, K, C
        _, visual_embeddings = x[4]
        # at this phase the global embedding is not used.
        text_features = F.normalize(text_features, dim=-1)  # t_bar
        visual_embeddings = F.normalize(visual_embeddings, dim=1)  # z_bar
        # 两个normalize之后得到的就是相当于概率分布的embedding
        score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        # BCHW * BKC -> BKHW 导致最后是 batch * k categories * height * width.
        # 也就是说明 对于每一个类别都有一个分割的图片出来
        score_map0 = F.upsample(score_map3, x[0].shape[2:], mode='bilinear')
        # 最后是输出score_map[0](是resolution比较大的那个map)
        # upsample the score map to the shape of 1st stage of backbone(higher resolution)
        #@TODO: considering if the score_map3 have some usages? e.g. serving as the quasi center point?
        score_maps = [score_map0, None, None, score_map3]
        return score_maps

    def compute_text_features(self, x, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone,
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x[4]
        # here global_feat is original x[4] passing doing GAP,which contains x_global and x_local
        B, C, H, W = visual_embeddings.shape
        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.class_names), C, device=global_feat.device)
        # dummy for FLOPs computation
        else:
            text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # print(visual_context.shape, visual_embeddings.shape, text_embeddings.shape)
        # using the base text_encoder to get text embeddings

        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff
        # t <-- t + gamma * Vpost, each time this class is called, one time of optimization is finished
        return text_embeddings

#@TODO:rewrite the dummy forward function to compare the FLOPs
    def forward_dummy(self, img):
        x = self.extract_feat(img)
        if self.use_seg_loss:
            x, score_map = x
        return

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_true_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
                Args:
                    img (Tensor): Input images of shape (N, C, H, W).
                        Typically these should be mean centered and std scaled.
                    img_metas (list[dict]): A List of image info dict where each dict
                        has: 'img_shape', 'scale_factor', 'flip', and may also contain
                        'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                        For details on the values of these keys see
                        :class:`mmdet.datasets.pipelines.Collect`.
                    gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                        image in [tl_x, tl_y, br_x, br_y] format.
                    gt_labels (list[Tensor]): Class indices corresponding to each box
                    gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                        boxes can be ignored when computing the loss.
                Returns:
                    dict[str, Tensor]: A dictionary of loss components.
                """
        x = self.extract_feat(img, use_seg_loss=self.use_seg_loss, use_pt_ambience=self.use_pt_ambience)
        if self.use_seg_loss or self.use_pt_ambience:
            x, score_map = x

        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]

        for stage in range(self.num_stages):
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                neg_proposal_list, neg_weight_list = None, None
                pseudo_boxes = generate_proposals
            else:
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas,
                                                                                   stage=stage)
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
            # The bbox_head is used here in the roi_head forward train.
            roi_losses, pseudo_boxes, dynamic_weight = self.roi_head.forward_train(stage, x, img_metas,
                                                                                   pseudo_boxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_true_bboxes, gt_labels,
                                                                                   dynamic_weight,
                                                                                   gt_bboxes_ignore, gt_masks,
                                                                                   **kwargs)
            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

            if self.use_seg_loss and stage==self.num_stages - 1:
                losses['loss_seg'] = self.compute_seg_loss(img, score_map, img_metas, pseudo_boxes, gt_labels)
            if self.use_pt_ambience and stage==self.num_stages - 1:
                losses['loss_amb'] = self.compute_pt_amb_loss(img, score_map, img_metas, gt_points, gt_labels, self.gen_range)
        return losses

#@TODO: change it in the form of detailed.pt
    def compute_seg_loss(self, img, score_map, img_metas, gt_bboxes, gt_labels):
        target, mask = self.build_seg_target(img, img_metas, gt_bboxes, gt_labels)
        loss = F.binary_cross_entropy(F.sigmoid(score_map + 1e-6), target, weight=mask, reduction='sum')
        # here the parameter 'weight=mask' is used to discard the background's BCE computation.
        loss = loss / mask.sum()
        return loss

#@TODO: come up with a method or a named module to refine the predicted pseudo box through the score map
    def compute_pt_amb_loss(self, img, score_map, img_metas, gt_points, gt_labels, gen_range, lambdap):
        # computing the correlation center point
        loc_map, mask = self.build_point_ambience(img, img_metas, gt_points, gt_labels, gen_range)
        loss = F.binary_cross_entropy(F.sigmoid(score_map), loc_map, weight=mask, reduction='sum')
        loss = loss / mask.sum() * lambdap
        return loss

    def build_point_ambience(self, img, img_metas, gt_points, gt_labels, gen_range):
        '''

        :param img:
        :param img_metas:
        :param gt_points:
        :param gt_labels:
        :param gen_range: the reliable focusing range of point supervision shouldn't be too large.
        :return:
        '''
        assert gen_range is int, 'the gen_range should be a integer'
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        gen_range //= 8
        loc_map = torch.zeros(B, len(self.class_names), H, W)
        mask =  torch.zeros(B, 1, H, W)
        for i, (gt_point_, gt_label_) in enumerate(zip(gt_points, gt_labels)):
            for (gt_point, gt_label) in zip(gt_point_, gt_point_):
                xc = gt_point[0] // 4
                yc = gt_point[1] // 4
                xl = xc - gen_range if (xc - gen_range) > 0 else 0
                xr = xc + gen_range if (xc + gen_range) < W else W
                yd = yc - gen_range if (yc - gen_range) > 0 else 0
                yu = yc + gen_range if (yc + gen_range) < H else H
                loc_map[i, gt_label, yd : yu, xl : xr] = 1
                mask[i, :, yd: yu, xl: xr] = 1
            mask = mask.expand(-1, len(self.class_names), -1, -1)
            loc_map = loc_map.to(img.device)
            mask = mask.to(img.device)

            return loc_map, mask

    def build_seg_target(self, img, img_metas, gt_bboxes, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        # set a pseudo segmentation mask for seg loss computation
        target = torch.zeros(B, len(self.class_names), H, W)  # BKHW
        mask = torch.zeros(B, 1, H, W)  # foreground and background
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            # the value of 'bboxes' tensor is four coordinations, such that this line yields scaling
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            # clamping the bbox coordinations to valid sizes
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                # setting the pixels inside the bbox to 1 in their corresponding target channels.
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                # the mask only distinguish foreground or background.
        mask = mask.expand(-1, len(self.class_names), -1, -1)  # expanding to BKHW, the same as target
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask

    def simple_test(self, img, img_metas, gt_bboxes, gt_anns_id, gt_true_bboxes, gt_labels,
                    gt_bboxes_ignore=None, proposals=None, rescale=False):
        """Test without augmentation."""
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        for stage in range(self.num_stages):
            gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
            else:
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas, stage=stage)

            test_result, pseudo_boxes = self.roi_head.simple_test(stage,
                                                                  x, generate_proposals, proposals_valid_list,
                                                                  gt_true_bboxes, gt_labels,
                                                                  gt_anns_id,
                                                                  img_metas,
                                                                  rescale=rescale)
        return test_result


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        if self.use_seg_loss:
            x, score_map = x
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )