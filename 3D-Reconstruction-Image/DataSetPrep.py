import os 
import urllib
from multiprocessing import Pool
from progress.bar import Bar
import sys
sys.path.insert(0, '../')
import scripts.binvox_rw
from scripts.global_variables import *
import numpy as np 
from tqdm import tqdm
from glob import glob
from datetime import datetime
import random 
import shutil
from PIL import Image, ImageOps
import scripts
import argparse

# this is the dataset for object translation, it will download the object files, convert then into numpy matricies, and overlay them onto pictures from the sun dataset 

parser = argparse.ArgumentParser(description='Dataset prep for image to 3D object translation, downloads and creates objects and image overlays.')
parser.add_argument('-o','--objects', default=['faces'], help='List of object classes to be used downloaded and converted.', nargs='+' )
parser.add_argument('-no','--num_objects', default=1000, help='number of objects to be converted', type = int)
parser.add_argument('-ni','--num_images', default=15, help='number of images to be created for each object', type = int)
parser.add_argument('-b','--backgrounds', default='sun/', help='location of the background images')
parser.add_argument('-t','--textures', default='dtd/', help='location of the textures to place onto the objects')
args = parser.parse_args()


#labels for the union of the core shapenet classes and the ikea dataset classes 
labels = {'03001627' : 'faces'}

# indicate here with set you want to use 
wanted_classes=[]
for l in labels: 
	if labels[l] in args.objects:
		wanted_classes.append(l)


debug_mode = 1 # change to make all of the called scripts print their errors and warnings 
if debug_mode:
	io_redirect = ''
else:
	io_redirect = ' > /dev/null 2>&1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'../'))

if not os.path.exists('data/voxels/'):
	os.makedirs('data/voxels/')
if not os.path.exists('data/objects/'):
	os.makedirs('data/objects/')

# these are two simple fucntions for parallel processing, down downloads in parallel, and call calls functions in parallel
# there work in conjuntion with pool.map()
def down(url):
	urllib.urlretrieve(url[0], url[1])
def call(command):
	os.system('%s %s' % (command, io_redirect))


def binvox(): # converts .obj files to .binvox files, intermidiate step before converting to voxel .npy files 
	for s in wanted_classes: 
		dirs = glob('data/objects/' + labels[s]+'/*.obj')
		commands =[]
		count = 0 
		for d in tqdm(dirs):
			command = './binvox -d 100 -pb ' + d # this executable can be found at http://www.patrickmin.com/binvox/ ,  -d 100 idicates resoltuion will be 100 by 100 by 100 , -pb is to stop the visualization
			commands.append(command)
			if count %200 == 0  and count != 0: #again parallelize to make quicker, be careful, while this runs your computer will not be useable!
				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
			count +=1 
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()


def convert(): # converts .binvox files to .npy voxels grids. A 5 times downsampling is also applied here. 
			   # I apply a downsampling instead of rendering the binvox file at 20 by 20 by 20 resolution as I found that 
			   # the binvoxer makes things to skinny and will often miss out sections of objects entirely if they are not large enough 
			   # to avoid this I render at  high resolution and then my downsampling 'encourages' all of the object to be seen at this resoltuion
	for directory in wanted_classes:
		directory = 'data/voxels/'+labels[directory]+'/' 
		if not os.path.exists(directory):
			os.makedirs(directory)

	dir_obj = ['/test', '/train', '/valid']
	for num in wanted_classes: 
		
		for dir in dir_obj:
			mods = glob('data/objects/'+labels[num]+dir+'/*.binvox')
			for m  in tqdm(mods):  
				with open(m , 'rb') as f:
					try: 
						model = scripts.binvox_rw.read_as_3d_array(f)
					except ValueError:
						continue
				
				# print("\nthis is model")
				# print(model)
				data = model.data.astype(int)
				
				down = 15 # how
				smaller_data = np.zeros([20,20,20])

				a,b,c = np.where(data==1)
				
				for [x,y,z] in zip(a,b,c): 
					count = 0
					br = False 
					u = x 
					uu=y
					uuu=z
					if x%2 ==1: u-=1
					if y%2 ==1: uu-=1
					if z%2 ==1: uuu-=1
					if smaller_data[u/down][uu/down][uuu/down]==1: continue 
					for i in range(down): 
						for j in range(down): 
							for k in range(down):
								try: 
									count += data[x+i][y+j][z+k] 
								except IndexError: 
									w = 0 
								if count >= 1: 
									if x%2 ==1: x-=1
									if y%2 ==1: y-=1
									if z%2 ==1: z-=1
									smaller_data[x/down][y/down][z/down]= 1 
									br = True 
									break 
							if br: break 
						if br: break 
				xx,yy,zz = np.where(smaller_data==1)
				if len(xx)<200: continue # this avoid objects whihc have been heavily distorted
				np.save( 'data/voxels/'+labels[num]+'/'+m.split('/')[-1][:-7], smaller_data) 				


def resize(): # for resizing images to 256 by 256  and zooming in on the objects 
	for s in wanted_classes: 
		files = glob('data/images/' + labels[s]+'/*') 
		for f in tqdm(files):
			im = Image.open(f)
			x,y = im.size
			im = ImageOps.fit(im, (256,256), Image.ANTIALIAS) # reshaping to 256 by 256 
			im.save(f)

def rename_files():
	path_ = "data/voxels/face"
	files = glob(path_ + '/*')
	for f in tqdm(files):
		actual = f
		new = f.split('full_object_')[0] + f.split('full_object_')[1]
		print("this is new")
		print(new)
		shutil.move(actual, new)
		

backgrounds = 'sun/' # location of desired background images, I just used the sun dataset 

print'resizing'
print'------------'
resize()
print'------------'
print'converting binvoxes from objects'
print'------------'
binvox()
print'------------'
print 'converting binvoxes to voxels'
print'------------'
convert()
print'------------'
print 'we are all done, Yay'


