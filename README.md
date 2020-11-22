# SiamGAT

## 1. Environment setup
This code has been tested on Ubuntu 16.04, Python 3.5, Pytorch 1.2.0, CUDA 9.0.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test
Download the pretrained model:  
[general_model](generalmodel_link) extract code:
[got10k_model](got10kmodel_link) extract code:
[lasot_model](lasotmodel_link) extract code:
 and put them into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

```bash 
python testTracker.py                                \
	--dataset UAV123                      \ # dataset_name
	--snapshot snapshot/general_model.pth  # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Evaluation
We provide the tracking [results](results_link) (extract code: xxxx) of GOT-10K, LaSOT, OTB100 and UAV123. If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV123                  \ # dataset_name
	--tracker_prefix 'general_model'   # tracker_name
```