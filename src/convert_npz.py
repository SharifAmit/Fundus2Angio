

import numpy as np
from numpy import asarray,savez_compressed
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import argparse

def convert_npz(imgpath,maskpath, size=(512,512),crops=50,n_images=17):
    src_list, tar_list = list(), list()
    for i in range(n_images):
        for j in range(crops):
            # load and resize the image
            filename = str(i+1)+"_"+str(j+1)+".png"
            mask_name = str(i+1)+"_mask_" + str(j+1)+".png"
            
            img = load_img(imgpath + filename, target_size=size)
            fundus_img = img_to_array(img)

            mask = load_img(maskpath + mask_name, target_size=size,color_mode="grayscale")
            angio_img = img_to_array(mask)
            
            # split into satellite and map
            src_list.append(fundus_img)
            tar_list.append(angio_img)
    return [asarray(src_list), asarray(tar_list)]
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--n_crops', type=int, default=50)
    parser.add_argument('--datadir', type=str, required=True, help='path/to/data_directory',default='data')
    parser.add_argument('--outfile_name', type=str, default='fun2angio')
    parser.add_argument('--n_images', type=int, default=17)
    args = parser.parse_args()

    # dataset path
    imgpath = args.datadir+'/Images/'
    maskpath = args.datadir+'/Masks/'
    # load dataset
    [src_images, tar_images] = convert_npz(imgpath,maskpath,size=(args.input_dim,args.input_dim),crops=args.n_crops,n_images=args.n_images)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # save as compressed numpy array
    filename = args.outfile_name+'.npz'
    savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)