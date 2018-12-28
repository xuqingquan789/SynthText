# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os
import sys
import traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
import pylab as pl
import scipy.io as sio
from scipy.stats import mode
import xml.etree.ElementTree as ET
from PIL import Image
# Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 10  # no. of times to use the same image
SECS_PER_IMG = 10  # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = '/workspace/wanghao/datasets/Synthtext_Original/data'
# DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/results2/new_synth.h5'
OUT_IMG = 'results/results2/train_images/'
OUT_MASK = 'results/results2/train_masks/'
OUT_GT ='results/results2/train_gts'

def get_data():
    """
    Download the image,depth and segmentation data:
    Returns, the h5 database.
    """
    if not osp.exists(DB_FNAME):
        try:
            colorprint(
                Color.BLUE, '\tdownloading data (56 M) from: '+DATA_URL, bold=True)
            print()
            sys.stdout.flush()
            out_fname = 'data.tar.gz'
            wget.download(DATA_URL, out=out_fname)
            tar = tarfile.open(out_fname)
            tar.extractall()
            tar.close()
            os.remove(out_fname)
            colorprint(Color.BLUE, '\n\tdata saved at:'+DB_FNAME, bold=True)
            sys.stdout.flush()
        except:
            print(colorize(Color.RED, 'Data not found and have problems downloading.', bold=True))
            sys.stdout.flush()
            sys.exit(-1)
    # open the h5 file and return:
    return h5py.File(DB_FNAME, 'r')


def parse_xml(xml_name):
    tree = ET.parse(xml_name)
    objs = tree.findall('object')
    box = []
    for i in range(len(objs)):
        bbox = objs[i].find('bndbox')
        x1 = float(bbox.find('xmin').text)-1
        y1 = float(bbox.find('ymin').text)-1
        x2 = float(bbox.find('xmax').text)-1
        y2 = float(bbox.find('ymax').text)-1
        box.append([x1, y1, x2, y2])
    return box


def parse_txt(txt_name):
    with open(txt_name) as f:
        tmp = f.readlines()
    tmp = [t.strip() for t in tmp]
    tmp = [t.replace('\xef\xbb\xbf', '') for t in tmp]
    tmp = [t.split(',') for t in tmp]

    box = []
    original_box = []
    for t in tmp:
        label = t[-1]
        if label == '###':
            continue
        tq = t[:-1]
        x = [int(l) for l in tq[0::2]]
        y = [int(m) for m in tq[1::2]]
        box.append([min(x), min(y), max(x), max(y)])
        original_box.append(np.asarray(list(zip(x, y))))
    return box, original_box

def save_txt(imgname,index,box,txt):
    prefix = imgname.split('.')[0]
    words = []
    for i in txt:
        words.extend(i.split())
    assert len(words)==box.shape[-1],'error, num words should match box'
    with open(osp.join(OUT_GT,prefix+'_%d.txt'%(index)),'w') as f:
        print(words)
        print(box)
        for i in range(len(words)):
            bb=box[:,:,i].T.reshape((-1,)) 
            line=(',').join(str(float(b)) for b in bb)
            line+=',%s'%(words[i])
            
            f.write(line+'\n')
def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        db['data'].create_group(dname)
        save_txt(imgname, i,res[i]['wordBB'],res[i]['txt'])
        save_img(imgname, i, res[i]['img'], res[i]['mask'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        L = res[i]['txt']
        L = [n.encode('ascii', 'ignore') for n in L]
        db['data'][dname].attrs['txt'] = L


def save_img(imgname, index, img, mask):
    prefix = imgname.split('.')[0]

    imname = prefix+'_%d.jpg' % (index)
    cv2.imwrite(osp.join(OUT_IMG, imname), img[:,:,::-1])
    cv2.imwrite(osp.join(OUT_MASK+imname), mask)


def main(viz=False):

    depth_db = h5py.File(osp.join(DATA_PATH, 'depth.h5'), 'r')
    seg_db = h5py.File(osp.join(DATA_PATH, 'seg.h5'), 'r')
    im_path = osp.join(DATA_PATH, 'bg_img')
    img_names = os.listdir(im_path)

    N = len(img_names)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN, 'Storing the output in: '+OUT_FILE, bold=True))

    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)
    for i in range(start_idx, end_idx):
        if i % 100 == 0:
            out_db.flush()
            os.system('cp %s %s_%d' % (OUT_FILE, OUT_FILE, i))
        try:
            # get the image:
            imname = img_names[i]
            img = osp.join(im_path, imname)
            img=Image.open(img)

            depth = depth_db[imname][:].T
            depth = depth[:, :, 1]

            seg = seg_db['mask'][imname][:].astype('float32')
            area = seg_db['mask'][imname].attrs['area']
            label = seg_db['mask'][imname].attrs['label']
            sz = depth.shape[:2][::-1]

            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
            depth = np.array(Image.fromarray(depth).resize(sz, Image.NEAREST))

            print(colorize(Color.RED, '%d of %d' % (i, end_idx-1), bold=True))
            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE, viz=viz)
            if len(res) > 0:
                add_res_to_db(imname, res, out_db)
            # visualize the output:
            if viz:
                if 'q' in raw_input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            # break
            continue
    out_db.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz',
                        default=False, help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
