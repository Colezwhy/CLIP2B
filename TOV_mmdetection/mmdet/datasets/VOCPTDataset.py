from collections import OrderedDict
import mmcv
from mmcv.utils import print_log
import os.path as osp
from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import os

@DATASETS.register_module()
class VOCPTDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, istrain=False, **kwargs):
        super(VOCPTDataset, self).__init__(istrain, **kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
        self.istrain = istrain

    ######### function edited by colez. to adjust to Point annotations.
    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations_qc_pt', f'{img_id}.xml') if self.istrain else osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        true_bboxes = []
        anns_id = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            anns_id.append(img_id)
            # the difficult samples are considered together
            # difficult = obj.find('difficult')
            # difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # added by colez.
            point = obj.find('point')
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False

            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if point is not None:
                true_bbox = bbox
                pt_ann = obj.find('point')
                X = int(float(pt_ann.find('X').text))
                Y = int(float(pt_ann.find('Y').text))
                point_cor = [X - 6, Y - 6, X + 6, Y + 6]
                # the point anno in the format of list and to satisfy the p2b format
                # an offset 12 12 is added.
                bbox = point_cor
                true_bboxes.append(true_bbox)
            # if point is not None:
            #     bboxes.append(bbox)
            #     labels.append(label)
            # if difficult or ignore:
            #     bboxes_ignore.append(bbox)
            #     labels_ignore.append(label)
            # else:
            bboxes.append(bbox)
            labels.append(label)


        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)


        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            anns_id=np.array(anns_id, dtype=np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        if len(true_bboxes) > 0:
            ann['true_bboxes'] = true_bboxes  # added by colez.
        return ann


    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        if self.istrain:
            for img_id in img_ids:
                filename = f'JPEGImages/{img_id}.jpg'
                xml_path = osp.join(self.img_prefix, 'Annotations_qc_pt',
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                if size is not None:
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                else:
                    img_path = osp.join(self.img_prefix, 'JPEGImages',
                                        '{}.jpg'.format(img_id))
                    img = Image.open(img_path)
                    width, height = img.size
                data_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
        else:
            for img_id in img_ids:
                filename = f'JPEGImages/{img_id}.jpg'
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                if size is not None:
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                else:
                    img_path = osp.join(self.img_prefix, 'JPEGImages',
                                        '{}.jpg'.format(img_id))
                    img = Image.open(img_path)
                    width, height = img.size
                data_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

#@ TODO: turn the collected pseudo boxes into a XML file as the annotation
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 save_result_file=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        pth = save_result_file + '/' + str(self.year)
        if os.path.exists(save_result_file) is False:
            os.mkdir(save_result_file)
            print('index created successfully.')
            if os.path.exists(pth) is False:
                os.mkdir(pth)
                print('%d created' % self.year)
        for idx in range(len(self)):

            cur_result = results[idx]
            img_id = self.data_infos[idx]['id']
            xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml') # find each img in the annotation file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            file_name = root.find('filename').text
            file_number = file_name[0:-4]
            xml_name = file_number + '.xml'
            obj_list = root.findall('object')  # return a list containing all object infos.
            cls = 0
            ptr = 0
            for cls_res in cur_result:
                if cls_res is None:
                    cls += 1
                    continue
                else:
                    for det in cls_res:
                        obj = obj_list[ptr]
                        obj.find('name').text = self.CLASSES[cls]
                        obj.find('bndbox').find('xmin').text = '%s' % int(det[0])
                        obj.find('bndbox').find('ymin').text = '%s' % int(det[1])
                        obj.find('bndbox').find('xmax').text = '%s' % int(det[2])
                        obj.find('bndbox').find('ymax').text = '%s' % int(det[3])
                        ptr += 1
                    cls += 1
            tree.write(os.path.join(pth, xml_name))
            if idx % 500 == 0:
                print('%d xml files generated.' % idx)

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
