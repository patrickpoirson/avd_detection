#!/bin/bash

#mkdir data
#cd data
#mkdir avd
#cd avd

# need url to get part tars
#wget part1.tar
#wget part2.tar

tar -xvf part1.tar
tar -xvf part2.tar 
rm part*.tar

cp ../../info/* . 

echo 'doing resize'

for d in */; do 
        cd $d
  echo $d
  cd jpg_rgb
        for file in *.jpg; do 
                convert $file -resize 600x600 $file; 
        done
        cd ../
  cd ../
done

echo 'resizing done'

cd ../../
echo 'downloading vgg models'
mkdir models
cd models
mkdir VGGNet
cd VGGNet

wget http://cs.unc.edu/~wliu/projects/ParseNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
wget https://gist.githubusercontent.com/weiliu89/2ed6e13bfd5b57cf81d6/raw/758667b33d1d1ff2ac86b244a662744b7bb48e01/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt
cd ../../

