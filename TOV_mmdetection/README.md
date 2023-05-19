The CLIP2B code is in mmdet/models/detectors/CLIP2B.py
our GPUs: 2 * RTX3090

# On VOC2007 dataset:
 The CLIP2B model achieves 63.7mAP on VOC2007 test set, \
 Comparing to 61.2mAP by P2BNet.


# On COCO dataset:
 To be updated.

# Prerequisites
install environment following
```shell script
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# install the latest mmcv

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    ```

# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
```

```shell script
conda install scikit-image  # or pip install scikit-image
```



#  Prepare dataset COCO
1. download dataset to data/coco
2. generate point annotation or download point annotation(
move annotations/xxx to data/coco/annotations_qc_pt/xxx

# Prepare dataset VOC
1. download dataset to data/VOC
2. downloading the point annotations on network disks.




### Train 
```open to the work path: CLIP2B/TOV_mmdetection```
1. CLIP2B + FasterRCNN
    ```shell script
    # [cmd 0] train CLIP2B and inference on training set with P2BNet
	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/COCO/CLIP2B/CLIP2B_r50_fpn_1x_coco_ms.py 2 \
	--work-dir=${work_dir}  \
	--cfg-options evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
	
    # [cmd 1] turn result file to coco annotation fmt
	python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dir/coco/_1200_latest_result.json  ../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json
    
    # [cmd 2] train FasterRCNN
    	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 \
	--work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
    ```

2. inference with trained CLIP2B to get pseudo box and train FasterRCNN with pseudo box
    ```shell script
    # [cmd 0] inference with trained P2BNet to get pseudo box
	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/COCO/CLIP2B/CLIP2B_r50_fpn_1x_coco_ms.py 2 \
	--work-dir=${work_dir}  \
	--cfg-options  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' load_from=${work_dir}'P2BNet/TOV_mmdetection_cache' evaluation.do_first_eval=True runner.max_epochs=0 
	
    # [cmd 1] turn result file to coco annotation fmt
	python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dir/coco/_1200_latest_result.json  ../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json
    
    # [cmd 2] train FasterRCNN
    	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 \
	--work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
    ```











