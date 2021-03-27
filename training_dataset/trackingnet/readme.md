# Preprocessing TrackingNet
TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild

### Prepare dataset

After download the dataset, please unzip the dataset at *train_dataset/TrackingNet* directory
````shell
mkdir data
unzip TrackingNet/zip/*.zip -d ./data
````

### Crop & Generate data info

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
