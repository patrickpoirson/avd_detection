import argparse
import os 
import os.path as osp
import subprocess
import sys
import random

from utils import options
from utils import makeAvdAnns
from examples.ssd import ssd_Avd

DATA_DIR = 'data/avd/'

def main(args):

    data_root_dir = DATA_DIR
    opt_dir = args['opt']

    if args['idx'] != 0 and opt_dir == '':
        # TODO remove 
        opt_dir = osp.join('/home/poirson/rohit/options/', str(args['idx']) + '.json')

    opt = options.Options(opt_dir)
    set_opts(args, opt)
    
    anns = makeAvdAnns.MakeAnns(opt)
    anns.run_main()
        
    mapfile = osp.join(DATA_DIR, 'labelmap%d.prototxt' % opt.get_opts('split_id'))
    anno_type='detection'
    label_type='json'
    split_id = opt.get_opts('split_id')

    if not osp.exists(osp.join(DATA_DIR, 'labelmap%d.prototxt' % split_id)):
        cmd = './build/tools/create_label_map --delimiter=" "\
                --include_background=true %s %s' \
                % (osp.join(DATA_DIR, 'map%d.txt' % split_id),\
                osp.join(DATA_DIR, 'labelmap%d.prototxt' % split_id))
        print cmd
        subprocess.call(cmd, shell=True)

    trstem = opt.get_avd_db_stem('train') 
    trdb = ('%s_lmdb' % trstem, trstem, 'train.txt')

    testem = opt.get_avd_db_stem('test') 
    testdb = ('%s_lmdb' % testem, testem, 'test.txt')


    with open(osp.join(data_root_dir, 'cache', testem, 'test.txt'), 'r') as infile:
        numTest = len([line for line in infile]) 
        opt.add_kv('num_test', numTest)


    splits = [trdb, testdb]

    for split in splits:
        listFile = osp.join(DATA_DIR, 'cache', split[1], split[2])
        outFile = osp.join(DATA_DIR, 'lmdb', split[0])

        cmd = 'python scripts/create_annoset.py --anno-type=%s --label-type=%s \
        --label-map-file=%s --encode-type=jpg --root=%s \
        --listfile=%s --outdir=%s' % \
        (anno_type, label_type, mapfile, './', listFile, outFile)

        print cmd
        subprocess.call(cmd, shell=True)

    mod_id = random.randint(1, 999000)
    if args['idx'] != 0:
        mod_id = args['idx']

    opt.add_kv('mod_id', mod_id)

    # hack
    f = open(osp.join(DATA_DIR, 'map%d.txt' % opt.get_opts('split_id') ))
    lines = [line for line in f]
    f.close()
    opt.add_kv('num_classes', len(lines) + 1)

    # TODO
    opt_out_path = osp.join('/home/poirson/options', '%d.json' % mod_id)
    opt.write_opt(opt_out_path)

    ssd = ssd_Avd.SSD()
    ssd.run_main(opt, data_root_dir)


def set_opts(args, opt):
    for k, v in args.iteritems():
        if k == 'opt' or k == 'idx':
            continue
        
        if k == 'gpu' and v != '':
            opt.add_kv(k, v)
        elif k != 'gpu' and v != -1:
            opt.add_kv(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for SSD on avd")

    parser.add_argument('--opt', default='', type=str, help='path to json file with options')
    parser.add_argument('--idx', default=0, type=int, help='specify model id to resume')
    parser.add_argument('--gpu', default='', type=str, help='specify which gpus to use')

    # experiment options
    parser.add_argument('--diff_max', default=-1, type=int, help='max diff to use')
    parser.add_argument('--split_id', default=-1, type=int, help='which split to use')
    parser.add_argument('--size', default=-1, type=int, help='im resolution')
    
    # likely leave fixed
    parser.add_argument('--step_size', default=-1, type=int, help='model step size')
    parser.add_argument('--max_iter', default=-1, type=int, help='max num iters')

    args = parser.parse_args()
    params = vars(args)
    main(params)
