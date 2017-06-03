## Installation

```
pip install visdom
pip install dominate
```

## Data

+ `datasets/girls/train`

+ `dataset/girls/val`


## Training

+ From scratch:

```
python train.py --dataroot ./datasets/girls --name girls_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --align_data --no_lsgan --use_dropout --batchSize 12 --save_latest_freq 2000 --niter 15 --niter_decay 15
```

+ Continue:

```
python train.py --dataroot ./datasets/girls --name girls_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --align_data --no_lsgan --use_dropout --batchSize 12 --save_latest_freq 2000 --niter 15 --niter_decay 15 --continue_train
```

## Generate

```
python gen.py --dataroot ./datasets/girls --name girls_pix2pix --model one_direction_test --which_model_netG unet_256 --which_direction AtoB --real_A real_A.jpg --fake_B fake_B.jpg
```

## Server

```
export FLASK_APP=server.py
flask run -host 0.0.0.0
```

or simpler with debug mode:

```
python server.py
```

Server will run at port 5000.
