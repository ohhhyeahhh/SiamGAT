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
        <td colspan="2" align=center> Dataset</td>
        <td align=center>SiamGAT</td>
        <td align=center>SiamGAT*</td>
        <td align=center>SiamGAT Model Link</td>
        <td align=center>SiamGAT* Model Link</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>GOT10k</td>
        <td>AO</td>
        <td>63.1</td> <td>67.1</td>
        <td rowspan="3" align=center>
			<a href="https://drive.google.com/file/d/1g4ETsJF_jtvpn0-6XF0VGxCtg67-EFmc/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/1wap-r-57Rl9NGndiSNAfHw">BaiduYun</a>(zktx)
		</td>
        <td rowspan="3" align=center>
			<a href="https://drive.google.com/file/d/1RiHKQzxt6MNJ3urMFI-J5CTaYnHLmuNa/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/1P95mpvJGoxJ1KW8EUgfiEQ">BaiduYun</a>(d74o)
		</td>
    </tr>
    <tr>
        <td>SR0.5</td>
        <td>74.6</td> <td>78.7</td>
    </tr>
    <tr>
        <td>SR0.75</td>
        <td>50.4</td> <td>58.9</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>TrackingNet</td>
        <td>Success</td>
        <td>75.3</td> <td>76.9</td>
        <td rowspan="3" align=center>
			<a href="https://drive.google.com/file/d/1D2FSYDepz8LU0D3ZsWPYdEIVpNItgHwl/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/1Zst1o1cg_zK9YqN3meJ7Bw">BaiduYun</a>(n2sm)
		</td>
        <td rowspan="9" align=center>
			<a href="https://drive.google.com/file/d/1WgZwzKzxz_qgFke8kY4UspCdXjHYxIPG/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/1WyuNhoyqJqBEzDdbMVFrRA">BaiduYun</a>(fxo2)
		</td>
    </tr>
    <tr>
        <td>Norm precision</td>
        <td>80.7</td> <td>82.4</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>69.8</td> <td>71.9</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>LaSOT</td>
        <td>Success</td>
        <td>53.9</td> <td> 59.5 </td>
        <td rowspan="3" align=center>
			<a href="https://drive.google.com/file/d/167ANy1557rcIsAjuH6_bSS_OFEvgG93s/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/17-pG-Mytg4sT330mhd584A">BaiduYun</a>(dilp)
		</td>
    </tr>
    <tr>
        <td>Norm precision</td>
        <td>63.3</td> <td> 69.0 </td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>53.0</td> <td> 61.2 </td>
    </tr>
        <tr>
        <td rowspan="3" align=center>VOT2020</td>
        <td>EAO</td>
        <td>-</td> <td> 0.453 </td>
        <td rowspan="3" align=center>-</td>
    </tr>
    <tr>
        <td>A</td>
        <td>-</td> <td> 0.756 </td>
    </tr>
    <tr>
        <td>R</td>
        <td>-</td> <td> 0.729 </td>
        </tr>
    <tr>
        <td rowspan="2" align=center>OTB100</td>
        <td>Success</td>
        <td>71.0</td> <td>71.5</td>
        <td rowspan="4" align=center>
			<a href="https://drive.google.com/file/d/1LKU6DuOzmLGJr-LYm4yXciJwIizbV_Zf/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/1nuK-gAX12K96CQpHbHr3tA">BaiduYun</a>(w1rs)
		</td>
        <td rowspan="2" align=center>
			<a href="https://drive.google.com/file/d/1JX7j93R5tQkfxC2NHHUkoIpE2dVGrMe-/view?usp=sharing">Google Driver</a>/<br>
			<a href="https://pan.baidu.com/s/1D_hrPpOPNcFYzaPbAINi_g">BaiduYun</a>(c6c5)
		</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>91.7</td> <td>93.0</td>
    </tr>
    <tr>
        <td rowspan="2" align=center>UAV123</td>
        <td>Success</td>
        <td>64.6</td> <td> - </td>
        <td rowspan="2" align=center>-</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>84.3</td> <td> - </td>
    </tr>
</table>

### Prepare testing datasets
Download testing datasets and put them into `test_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

### Test the tracker
```bash 
python testTracker.py \    
        --config ../experiments/siamgat_googlenet_ct_alldataset/config.yaml \ # siamgat_xx_xx for SiamGAT, siamgat_ct_xx_xx for SiamGAT*
	--dataset OTB100 \                                 # dataset_name： GOT-10k, LaSOT, TrackingNet, OTB100, UAV123
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
    --cfg ../experiments/siamgat_googlenet/config.yaml # siamgat_xx_xx for SiamGAT, siamgat_ct_xx_xx for SiamGAT*
```

## 4. Evaluation

We provide tracking results for comparison: 
- SiamGAT: [BaiduYun](https://pan.baidu.com/s/1HBE0Kn2ietvQT7NLExAQoA) (extract code: 8nox) or [GoogleDriver](https://drive.google.com/file/d/1xAbTfJNKpGJykdFrTDGtHQtxgKHOhgL7/view?usp=sharing).
- SiamGAT*: [BaiduYun](https://pan.baidu.com/s/1dWhUxsJyE37d8PfOdqFR_g) (extract code: kjym) or [GoogleDriver](https://drive.google.com/file/d/19nzlqz9aCswQwnnvc9AS7btAg_uLCTYI/view?usp=sharing).

If you want to evaluate the tracker on OTB100, UAV123 and LaSOT, please put those results into `results` directory and then run `eval.py` . 
Evaluate GOT-10k on [Server](http://got-10k.aitestunion.com/). Evaluate TrackingNet on [Server](https://tracking-net.org/).  

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

> @article{cui2022joint,  
	title={Joint Classification and Regression for Visual Tracking with Fully Convolutional Siamese Networks},  
	author={Cui, Ying and Guo, Dongyan and Shao, Yanyan and Wang, Zhenhua and Shen, Chunhua and Zhang, Liyan and Chen, Shengyong},  
	journal={International Journal of Computer Vision},  
	year={2022},  
	publisher={Springer},  
	doi = {10.1007/s11263-021-01559-4}  
}

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
