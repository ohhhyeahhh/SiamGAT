# SiamGAT

## 1. Environment setup
This code has been tested on Ubuntu 16.04, Python 3.5, Pytorch 1.2.0, CUDA 9.0.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test
Download the pretrained model and put them into `tools/snapshot` directory.   
From BaiduYun:
* [otb_uav_model](https://pan.baidu.com/s/1nuK-gAX12K96CQpHbHr3tA) extract code: w1rs  
* [got10k_model](https://pan.baidu.com/s/1LcKRO4t3vqGs8r7Lb73lmA) extract code: n91w  
* [lasot_model](https://pan.baidu.com/s/17-pG-Mytg4sT330mhd584A) extract code: dilp  
* [TrackingNet_model](https://pan.baidu.com/s/1Zst1o1cg_zK9YqN3meJ7Bw) extract code: n2sm  
  
From Google Driver:
* [otb_uav_model](https://drive.google.com/file/d/1LKU6DuOzmLGJr-LYm4yXciJwIizbV_Zf/view?usp=sharing)
* [got10k_model](https://drive.google.com/file/d/1f0wZXMnzIOIWTTtL7D_Z7N42FAzY8sDi/view?usp=sharing)
* [lasot_model](https://drive.google.com/file/d/167ANy1557rcIsAjuH6_bSS_OFEvgG93s/view?usp=sharing)


Download testing datasets and put them into `test_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

The tracking results can be download from [BaiduYun](https://pan.baidu.com/s/1Ohit3C_hdy70x-JrdGUfeg) (extract code: 0wod) or [GoogleDriver](https://drive.google.com/file/d/1GBk_eKOMxo3rdTrmZYzDaG-Nc_j2Cdg6/view?usp=sharing) for comparision.

```bash 
python testTracker.py \    
        --config ../experiments/siamgat_googlenet_otb_uav/config.yaml \
	--dataset UAV123 \                                 # dataset_name
	--snapshot snapshot/otb_uav_model.pth              # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train

### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://pan.baidu.com/s/1gQKmi7o7HCw954JriLXYvg) (code: v7s6)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)
* [LaSOT](https://cis.temple.edu/lasot/)

**Note:** `training_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

### Download pretrained backbones
Download pretrained backbones from [link](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) and put them into `pretrained_models` directory.

### Train a model
To train the SiamGAT model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

## 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1Ohit3C_hdy70x-JrdGUfeg) (extract code: 0wod) ([results in Google driver](https://drive.google.com/file/d/1GBk_eKOMxo3rdTrmZYzDaG-Nc_j2Cdg6/view?usp=sharing)) of GOT-10k, LaSOT, OTB100 and UAV123. TrackingNet results can be obtained from [BaiduYun](https://pan.baidu.com/s/1cJkTbhO73KaIfBzFHkonNg) (extract code: iwlj).If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV123                  \ # dataset_name
	--tracker_prefix 'otb_uav_model'   # tracker_name
```

## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot) and [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR). We would like to express our sincere thanks to the contributors.

## 6. Cite
If you use SiamGAT in your work please cite our papers:

> @InProceedings{Guo_2021_CVPR,  
  author = {Guo, Dongyan and Shao, Yanyan and Cui, Ying and Wang, Zhenhua and Zhang, Liyan and Shen, Chunhua},  
  title = {Graph Attention Tracking},  
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
  month = {June},  
  year = {2021}  
}

> @InProceedings{Guo_2020_CVPR,  
   author = {Guo, Dongyan and Wang, Jun and Cui, Ying and Wang, Zhenhua and Chen, Shengyong},  
   title = {SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking},  
   booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
   month = {June},  
   year = {2020}  
}
