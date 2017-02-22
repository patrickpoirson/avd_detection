#!/bin/bash
#cd images
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

