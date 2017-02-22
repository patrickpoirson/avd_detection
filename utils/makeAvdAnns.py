import numpy as np
import sys
import os.path as osp
import os
import scipy.io as sio
import json
import random
import argparse
import shutil
from random import shuffle
from utils import options



'''
This script preprocesses the annotations for active vision dataset
''' 


DATA_DIR = 'data/avd/'

class MakeAnns:

    def __init__(self, opt):
        self.opt = opt

    def run_main(self):

        base_path = DATA_DIR

        split_id = self.opt.get_opts('split_id')
        diff_max = self.opt.get_opts('diff_max')

        splits = getSplit(split_id)
        
        anns = getAnns(splits)
        anns = cleanAnns(anns, diff_max)

        anns['train'] = remEmpty(anns['train'])
        anns['test'] = remEmpty(anns['test'])

        idx2lbl = readLabelToCls()
        cls_present = getAllClasses(anns, idx2lbl)
        anns = updateLabels(anns, idx2lbl, cls_present)
        
        lblmap_name = osp.join(base_path, 'map%d.txt' % split_id)
        if not osp.exists(lblmap_name):
            makeLabelMap(cls_present, lblmap_name)

        train_dir = self.opt.get_avd_db_stem('train')
        tes_dir = self.opt.get_avd_db_stem('test')

        if not osp.exists(osp.join(base_path, 'cache/')):
            os.mkdir(osp.join(base_path, 'cache/'))

        splits = [train_dir, tes_dir]
        for split in splits:
            if not osp.exists(osp.join(base_path, 'cache', split)):
                os.mkdir(osp.join(base_path, 'cache', split))

        tr = False
        test = False

        if not osp.exists(osp.join(base_path, 'cache', train_dir, 'train.txt')):
            tr = True
            trList = []

        if not osp.exists(osp.join(base_path, 'cache', tes_dir, 'test.txt')):
            teList = []
            test = True

        for split, data in anns.iteritems():
            for idx, ann in data.iteritems():
                if split == 'train':
                    annpath = osp.join(base_path, 'cache', train_dir)
                elif split == 'test':
                     annpath = osp.join(base_path, 'cache', tes_dir)

                annLoc = osp.join(annpath, idx[:-4] + '.json')
                output = getImPath(ann, idx) + ' ' + annLoc + '\n'
                
                if not osp.exists(getImPath(ann, idx)):
                    print '%s does not exist ' % getImPath(ann, idx) 
                    sys.exit()

                json.dump(ann, open(annLoc, 'w'))

                if split == 'train' and tr:
                    trList.append(output)
                elif split == 'test' and test:
                    teList.append(output)

        if tr:
            with open(osp.join(base_path, 'cache', train_dir, 'train.txt'), 'w') as outfile:
                shuffle(trList)
                for line in trList:
                    outfile.write(line)

        if test:
            with open(osp.join(base_path, 'cache', tes_dir, 'test.txt'), 'w') as outfile:
                shuffle(teList)
                for line in teList:
                    outfile.write(line)


def makeLabelMap(cls_present, fi_name):
    f = open(fi_name, 'w')
    for i, cls_name in enumerate(cls_present):
        f.write('%s %d %s\n' % (cls_name, i+1, cls_name)) 
    f.close()


def updateLabels(data, idx2lbl, cls_present):
    for ann in data.itervalues():
        for im in ann.itervalues():
            for obj in im['annotation']:
                cat_id = obj['category_id']
                cls_name = idx2lbl[cat_id]
                obj['category_id'] = cls_name
    return data


def getAllClasses(data, id2cls):
    classes = []

    for split in data.itervalues():
        for im in split.itervalues():
            for obj in im['annotation']:
                cls_name = id2cls[obj['category_id']]
                if cls_name not in classes: classes.append(cls_name)
    return classes


def getImPath(ann, idx):
    path = osp.join(DATA_DIR, ann['scene_name'], 'jpg_rgb', idx)
    return path


def getSplit(split_id):
    fi_name = osp.join(DATA_DIR, 'split%d.txt' % split_id)
    if not osp.exists(fi_name):
        print '%s not found. exiting'
        sys.exit()

    f = open(fi_name, 'r')
    lines = [line.strip('\n') for line in f]
    f.close()

    tr_split = lines[0].split(' ')
    te_split = lines[1].split(' ')
    
    out = {}
    out['train'] = tr_split
    out['test'] = te_split
    return out 


def getAnns(splits):
    all_anns = {}

    for split in splits.iterkeys():
        all_anns[split] = {}
        for sc in splits[split]:
             jsfi = json.load(open(osp.join(DATA_DIR, sc, 'annotations.json'), 'r'))
             
             for k in jsfi.iterkeys(): jsfi[k]['scene_name'] = sc

             all_anns[split].update(jsfi)

    return all_anns


def readLabelToCls():
    cls2lbl = {}
    with open(osp.join(DATA_DIR, 'labels.txt'), 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            cls_name, idx = line.split(' ')
            cls2lbl[int(idx)] = str(cls_name)
    return cls2lbl


def remEmpty(data):
    keysToRemove = []
    for k, v in data.iteritems():
        if len(v['annotation']) == 0: keysToRemove.append(k)

    for k in keysToRemove:
        del data[k]
    return data


def cleanAnns(anns, diff_max):
    scale_x = (600.0/1920.0)
    scale_y = (338.0/1080.0)

    for ann_spli in anns.itervalues():
        for imid in ann_spli.itervalues():
            imid['image'] = {}
            imid['image']['width'] = int(600)
            imid['image']['height'] = int(338)

            list_obj = []
            for box in imid['bounding_boxes']:
                objects = {}
                if box[5] > diff_max:
                    continue 

                cat_id = box[4]
                objects['category_id'] = cat_id
                
                bbox = box[:4]
                bbox[0], bbox[1] = float(bbox[0])*scale_x, float(bbox[1])*scale_y
                bbox[2], bbox[3] = float(bbox[2])*scale_x, float(bbox[3])*scale_y

                # Convert to w,h
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]

                objects['bbox'] = bbox
                list_obj.append(objects)

            imid['annotation'] = list_obj

    return anns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_id', type=int, default=1, 
                        help='int id of the scene to use for testing')
    parser.add_argument('--diff_max', type=int, default=3, 
                        help='int id of the scene to use for testing')

    args = parser.parse_args()
    params = vars(args) # turn into a dict

    a = MakeAnns(params)
    a.run_main()
