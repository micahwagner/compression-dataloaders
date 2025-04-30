How to run:

CIFAR-10:
Inside of /cifar, create a directory called 'data'. 
Place the contents of the CIFAR-10 dataset in cifar/data/ (batches.meta, data_batach_1, ...).
In cifar.py, go to main(). The jpeg variable on line 78 controls whether or not compression is on.
From the root directory, you can now run cifar.py with the command: python cifar/cifar.py

ImageNet100:
Inside of /imagenet, create a directory called 'data'. 
Place the contents of the ImageNet100 dataset in imagenet/data/ (Labels.json, val.X, ...).
Rename val.X to 'valid'.
Inside of imagenet/data/, create a directory 'train'.
Place the contents of train.X- to imagenet/data/train.

To actually run offline compression, you must first run the command: python offline_compression.py
This only needs to be done once ever. It creates the imagenet/jpeg_data/ directory containing the compressed images.

In imagenet.py, go to main(). The jpeg variable on line 81 controls whether or not compression is on.
The offline variable on line 82 controls whether you are using the pre-compressed images. 
From the root directory, you can now run imagenet.py with the command: python imagenet/imagenet.py

Results will be placed in /results/.