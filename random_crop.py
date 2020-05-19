from PIL import Image
import numpy as np
import os
import random
import argparse

def random_crop(img, mask, height, width, num_of_crops,name,stride=1,dir_name='data'):
    Image_dir = dir_name + '/Images'
    Mask_dir = dir_name + '/Masks'
    directories = [dir_name,Image_dir,Mask_dir]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    max_crops = (max_x)*(max_y)

    crop_seq = [i for i in range(0,max_crops)]

    for i in range(num_of_crops):
        crop = random.choice(crop_seq)
        #print("crop_value for",i,":",crop)
        if crop ==0:
            x = 0
            y = 0
        else:
            x = int((crop+1)/max_y)
            #print(x)
            y = int((crop+1)%max_y)
            #print(y)
        crop_img_arr = img[x:x+width,y:y+height]
        #print(crop_img_arr.shape)
        crop_mask_arr = mask[x:x+width,y:y+height]
        crop_img = Image.fromarray(crop_img_arr)
        crop_mask = Image.fromarray(crop_mask_arr)
        img_name = directories[1] + "/" + name + "_" + str(i+1)+".png"
        mask_name = directories[2] + "/" + name + "_mask_" + str(i+1)+".png"
        crop_img.save(img_name)
        crop_mask.save(mask_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--n_crops', type=int, default=50)
    parser.add_argument('--datadir', type=str, required=True, help='path/to/data_directory',default='Dataset')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()

    # Abnormal Fundus/Angio Image pairs
    ab = [4,5,6,8,18,21,28]
    for i in range(7):
        img_name = args.datadir+"/ABNORMAL/"+str(ab[i])+".jpg"
        im = Image.open(img_name)
        img_arr = np.asarray(im)
        mask_name = args.datadir+"/ABNORMAL/"+str(ab[i])+"-"+str(ab[i])+".jpg"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask)
        name = str(i+1)
        random_crop(img_arr, mask_arr, args.input_dim, args.input_dim, args.n_crops, name)

    # Normal Fundus/Angio Image pairs
    n = [1,4,7,8,11,15,16,21,24,28]
    k=0
    for i in range(7,17):
        img_name = args.datadir+"/NORMAL/"+str(n[k])+"-"+str(n[k])+".jpg"
        im = Image.open(img_name)
        img_arr = np.asarray(im)
        mask_name = args.datadir+"/NORMAL/"+str(n[k])+".jpg"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask)
        name = str(i+1)
        random_crop(img_arr, mask_arr, args.input_dim, args.input_dim, args.n_crops,name,dir_name=args.output_dir)
        k=k+1