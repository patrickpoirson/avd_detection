# AVD Detection: Instance detection on Active Vision Dataset

### Introduction
We train an SSD detector for instances appearing in our Active Vision Dataset (AVD). 

### Citing

Please cite our dataset if it helps your research:

    @inproceedings{active-vision-dataset2017,
       author = {Ammirato, Phil and Poirson, Patrick and Park, Eunbyung and Kosecka, Jana and Berg, Alexander C.},
       title = {A Dataset for Developing and Benchmarking Active Vision},
       booktitle={IEEE International Conference on Robotics and Automation (ICRA)}, 
       year={2017} 
    }

Please also cite SSD:

    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/patrickpoirson/avd_detection.git
  cd avd_detection
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make py
  ```

### Preparation
1. Download dataset from (http://cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.html). We assume the tar files are stored in `./data/avd/`

2. Run setup script to prepare the data
  ```Shell
  ./setup.sh
  ```

### Train/Eval
1. Train your model with default settings.
  ```Shell
  # default call
  python driverAvd.py

  # for available options
  python driverAvd.py --h

  # Ex. train 512x512 model on second split using gpu 0 and 1
  python driverAvd.py --split_id 2 --gpu 0,1 --size 512 

  # Ex. evaluating trained model with id 4444
  python testAvd.py --model 4444 --iter 20000 --gpu 0,1

  ```