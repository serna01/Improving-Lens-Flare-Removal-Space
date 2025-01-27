I am finishing a programming job and took some notes from the most recent project i had, from this I have to derive 1 readme file for github, the main one is for ILF and at the end comment on the other models i tested, help me organize and create a very readable and didactic readme file for the github project, this are the notes: 3 models were tested:
7Kpp
Google research
Improved Lens Flare Removal (of google’s) (ILF)

Model​
Paper​
Github​
7Kpp​
Flare7K++: Mixing Synthetic and Real Datasets for Nighttime Flare Removal and Beyond​
https://github.com/ykdai/Flare7K.git​
Google Research​
How to train neural networks for flare removal. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.​
https://github.com/google-research/google-research/tree/master/flare_removal​
​
Improving lens flare removal​ (ILF)
Improving Lens Flare Removal with General Purpose Pipeline and Multiple Light Sources Recovery (ICCV 2023)​
https://github.com/YuyanZhou1/Improving-Lens-Flare-Removal


To test them I recommend using python virtual environments since each model has their own dependencies and different requirements.txt.
Also install NVIDIA CUDA for best results. In some cases it is even mandatory!

To run each of them:
Flare7kpp: https://colab.research.google.com/drive/1mQVum2Uy2fsl6l907uDmPWu4Ahm7JG9Y​

python test_large.py --input test/test_images/ --output result --model_path experiments/net_g_last.pth --flare7kpp​

Custom model:​

python test_large.py --input test/test_images/ --output result/customv2_6ch --model_path experiments/customv2_6ch/models/net_g_latest.pth --flare7kpp​

​

Google: ​

pip install tensorflow[and-cuda]==2.15.1​

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"​


python3 -m flare_removal.python.remove_flare   --ckpt=../train/model   --input_dir=../test/test_images   --out_dir=../results​


python3 -m flare_removal.python.remove_flare   --ckpt=../train_gpu/model   --input_dir=../test/test_images   --out_dir=../results_GPU​
​
--Improving lens flare (ILF)​

export PATH=/usr/local/cuda-12.5/bin:$PATH​

python3 remove_flare.py --input_dir=input_data/LL_lensflare --out_dir=results --model=Uformer --batch_size=2 --ckpt=experiments/trained_model​

Custom model:​

python3 remove_flare.py --input_dir=input_data/LL_lensflare --out_dir=results/model05092024 --model=Uformer --batch_size=2 --ckpt=experiments/ASM_trained_model05092024/model​

​

​

Jetson Nano Orin ( Jetpack6.0​ - Leo08)
Flare 7kpp --> Pytorch​

https://forums.developer.nvidia.com/t/torchvision-version-jetpack-6-0/301709/2​

https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/​


Google-research --> Tensorflow​

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev​

pip3 install cython​

pip3 install h5py​

pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/tensorflow-2.15.0+nv24.05-cp310-cp310-linux_aarch64.whl

Trainning models with custom data:

7Kpp​

python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml --debug​

python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml (by default 600K iters, which I think is too much, to check just in case)

Google-lens-flare-removal:​

python3 -m flare_removal.python.train \​
  --train_dir=/path/to/training/logs/dir \​
  --scene_dir=/path/to/flare-free/training/image/dir \​
  --flare_dir=/path/to/flare-only/image/dir​

​

python3 -m flare_removal.python.train \​
 --train_dir=../train/ \​
 --scene_dir= ../Flickr24K/(24Kimages) \​
 --flare_dir= ../test/lens-flare/(Captured2000 and simulated3000)​

​

How much time does this take on average load? 30 Mins for 95 images.​

Test:​

python3 -m flare_removal.python.train   --train_dir=../train/blender   --scene_dir=../blender_output_12k/batch_001/gt   --flare_dir=../test/lens-flare​

​

ILF
python train.py	  --flarec_dir=path/to/captured/flare   --flares_dir=path/to/simulated/flare    --scene_dir=path/to/scene/image​

Scenes need to be 640*640, flares 1008 752​

python resize_images.py /path/to/input_dir /path/to/output_dir 640 640 1008 752​

export PATH=/usr/local/cuda-12.5/bin:$PATH (To erase warning about ptxas)​


python train.py      --flarec_dir=test_data/Scene/real1008/input   --flares_dir=test_data/Flares/Flare-R/Light_Source752    --scene_dir=test_data/Scene/real640/gt​

To check progress, you can run on another terminal:

tensorboard --logdir=/tmp/train/summary​

tensorboard --logdir=~/GLF_removal/train_gpu/summary​

Core flares: 1400*1400 but neeeded in 1008*752 -->example (resize python code in utils):​

python resize_images.py test_data/Flares/Scattering_Flare/Core test_data/Flares/Scattering_Flare/Core752 1440 1440 1008 752​

python train.py      --flarec_dir=test_data/Flares/Flare-R/Light_Source752  --flares_dir=test_data/Flares/Scattering_Flare/Core752     --scene_dir=test_data/Scene/real640/gt​

​

Takes 12 mins​

python train.py --flarec_dir=test_data/Flares/Flare-R/Light_Source752 --flares_dir=test_data/batch_001/flares/752 --scene_dir=test_data/batch_001/gt (this one works but apparently the flares are being used stacked over the scene and sometimes rotated)​

To train with custom pipeline to use the simulation from dave where the scene already has the flare (scene_with_flare,  # Original scene with flare     -   pred_scene,      # Model prediction, ground truth)

SynthesisV2.py ​(new script)

python trainV2.py --flare_dir=test_data/batch_003/flares640 --scene_dir=test_data/batch_003/gt640​

python3 remove_flare.py --input_dir=input_data --out_dir=results/blender_modelv15102024 --model=Uformer --batch_size=2 --ckpt=experiments/blender_model/train/model
 
Notes:
In the data_provider.py code, the image shapes for the scene and flare datasets are specified in the following functions:​

Scene Dataset (get_scene_dataset):​

Default input shape: (640, 640, 3)​

This means scene images are expected to have a default size of 640x640 pixels.​

Flare Dataset (get_flare_dataset and get_flare_dataset2):​

Default input shape: (752, 1008, 3)​

This indicates that flare images are expected to be of size 752x1008 pixels.

python3 -m flare_removal.python.train \​
 --train_dir=../train/ \​
 --scene_dir= ../blender_output_12k/batch_001/gt/ \​
 --flare_dir= ../blender_output_12k/batch_001/flares/
