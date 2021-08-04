# SiamGAT

## 1. Environment setup
This code has been tested on Ubuntu 16.04, Python 3.5, Pytorch 1.2.0, CUDA 9.0.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test

<table>
    <tr>
        <td colspan="2" align=center> DATASET NAME</td>
        <td>SiamGAT</td>
        <td>model link</td>
    </tr>
    <tr>
        <td rowspan="2" align=center>OTB100</td>
        <td>Success</td>
        <td>0.71</td>
        <td rowspan="4" align=center>
			<a href="https://drive.google.com/file/d/1LKU6DuOzmLGJr-LYm4yXciJwIizbV_Zf/view?usp=sharing">Google Driver</a><br>
			<a href="https://pan.baidu.com/s/1nuK-gAX12K96CQpHbHr3tA">BaiduYun</a>(w1rs)
		</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>0.917</td>
    </tr>
    <tr>
        <td rowspan="2" align=center>UAV123</td>
        <td>Success</td>
        <td>0.646</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>0.843</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>LaSOT</td>
        <td>Success</td>
        <td>0.539</td>
        <td rowspan="3" align=center>
			<a href="https://drive.google.com/file/d/167ANy1557rcIsAjuH6_bSS_OFEvgG93s/view?usp=sharing">Google Driver</a><br>
			<a href="https://pan.baidu.com/s/17-pG-Mytg4sT330mhd584A">BaiduYun</a>(dilp)
		</td>
    </tr>
    <tr>
        <td>Norm precision</td>
        <td>0.633</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>0.53</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>GOT10k</td>
        <td>AO</td>
        <td>0.627</td>
        <td rowspan="3" align=center>
			<a href="https://drive.google.com/file/d/1f0wZXMnzIOIWTTtL7D_Z7N42FAzY8sDi/view?usp=sharing">Google Driver</a><br>
			<a href="https://pan.baidu.com/s/1LcKRO4t3vqGs8r7Lb73lmA">BaiduYun</a>(n91w)
		</td>
    </tr>
    <tr>
        <td>SR0.5</td>
        <td>0.743</td>
    </tr>
    <tr>
        <td>SR0.75</td>
        <td>0.488</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>TrackingNet</td>
        <td>Success</td>
        <td>75.26</td>
        <td rowspan="3" align=center>
			<a href="https://pan.baidu.com/s/1Zst1o1cg_zK9YqN3meJ7Bw">BaiduYun</a>(n2sm)
		</td>
    </tr>
    <tr>
        <td>Norm precision</td>
        <td>80.65</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>69.74</td>
    </tr>
</table>

Download testing datasets and put them into `test_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

The tracking result can be download from [BaiduYun](https://pan.baidu.com/s/1Ohit3C_hdy70x-JrdGUfeg) (extract code: 0wod) or [GoogleDriver](https://drive.google.com/file/d/1GBk_eKOMxo3rdTrmZYzDaG-Nc_j2Cdg6/view?usp=sharing) for comparision.

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
* [TrackingNet](https://tracking-net.org/#downloads)

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
We provide the tracking [results](https://pan.baidu.com/s/1Ohit3C_hdy70x-JrdGUfeg) (extract code: 0wod) ([results in Google driver](https://drive.google.com/file/d/1GBk_eKOMxo3rdTrmZYzDaG-Nc_j2Cdg6/view?usp=sharing)) of GOT-10k, LaSOT, OTB100 and UAV123. If you want to evaluate the tracker on OTB100, UAV123 and LaSOT, please put those results into  `results` directory. Evaluate GOT-10k on [Server](http://got-10k.aitestunion.com/).   
Get TrackingNet results from [BaiduYun](https://pan.baidu.com/s/1cJkTbhO73KaIfBzFHkonNg) (extract code: iwlj), and evaluate it on [Server](http://eval.tracking-net.org/).
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
