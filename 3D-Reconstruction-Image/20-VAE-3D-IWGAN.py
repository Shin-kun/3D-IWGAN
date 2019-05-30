import tensorflow as tf 
import os
import sys 
sys.path.insert(0, '../')
import tensorlayer as tl
import numpy as np
import random 
import scripts
from scripts.GANutils import *
from scripts.models import *
import argparse
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y%m%d%H%M%S',level=logging.DEBUG)

parser = argparse.ArgumentParser(description='3D-GAN implementation for 32*32*32 voxel output')
parser.add_argument('-i','--images', default='data/images/faces', help ='The location for the images.' )
parser.add_argument('-n','--name', default='LSGAN_64', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='data/voxels/face_64', help ='The location for the object voxel models.' )
parser.add_argument('-e','--epochs', default=4501, help ='The number of epochs to run for.', type=int)
parser.add_argument('-b','--batchsize', default=256, help ='The batch size.', type=int)
parser.add_argument('-sample', default= 5, help='How often generated obejcts are sampled and saved.', type= int)
parser.add_argument('-save', default= 50, help='How often the network models are saved.', type= int)
parser.add_argument('-graph', default= 5, help='How often the discriminator loss and the reconstruction loss graphs are saved.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= '', help='The epoch to number to be loaded from.', type=str)
parser.add_argument('-t', '--train', default=False, help='Indicates if model is already trained', action='store_true')
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name + "/"
save_dir =  "savepoint/" + args.name + "/"
output_size = 64

######### make directories ############################

make_directories(checkpoint_dir,save_dir)

####### inputs  ###################
images      = tf.placeholder(tf.float32, [args.batchsize, 256, 256,3 ], name='images')
real_models = tf.placeholder(tf.float32, [args.batchsize, output_size, output_size, output_size] , name='real_models')
z           = tf.random_normal((args.batchsize, 200), 0, 1)
eps         = tf.random_normal((args.batchsize, 200), 0, 1)
########## network computations #######################

net_m, net_s, means, sigmas = VAE(images) # means in the input vector, variance is used for error 
z_x = tf.add(means,  tf.multiply(sigmas, eps))

# this is for generating dcgan instead 
net_g, G_dec         = generator_64(z_x, batch_size=args.batchsize, is_train=True, reuse=False)
net_g2, G_train      = generator_64(z, batch_size=args.batchsize, is_train=True, reuse=True)

net_d, D_dec_fake    = discriminator_64(G_dec, output_size, batch_size= args.batchsize, is_train = True, reuse= False)
net_d2, D_fake       = discriminator_64(G_train, output_size, batch_size= args.batchsize, is_train = True, reuse= True)
net_d2, D_legit      = discriminator_64(real_models, output_size, batch_size= args.batchsize, is_train= True, reuse = True)
# net_eval, D_eval      = discriminator_DCGAN(real_models,  output_size, batch_size= args.batchsize, is_train= False, reuse = True) # this is for desciding weather to train the discriminator

# Comment out in order to train DC-GAN
#net_g, G_dec        = generator_20(z_x, batch_size= args.batchsize, is_train=True, reuse = False)
#net_g2, G_train     = generator_20(z, batch_size= args.batchsize, is_train = True, reuse=True)

#net_d, D_dec_fake   = discriminator(G_dec, output_size, batch_size= args.batchsize, improved = True ,is_train = True, reuse= False)
#net_d2 , D_fake     = discriminator(G_train, output_size, batch_size= args.batchsize, improved = True, is_train = True, reuse = True)
#net_d2, D_legit     = discriminator(real_models,  output_size, batch_size= args.batchsize, improved = True, is_train= True, reuse = True)

########## Gradient penalty calculations ##############
# alpha               = tf.random_uniform(shape=[args.batchsize,1] ,minval =0., maxval=1.)
# difference          = G_train - real_models
# inter               = []
# for i in range(args.batchsize): 
#     inter.append(difference[i] *alpha[i])
# inter = tf.unstack(inter)
# interpolates        = real_models + inter
# gradients           = tf.gradients(discriminator(interpolates, output_size, batch_size= args.batchsize, improved = True, is_train = False, reuse= True)[1],[interpolates])[0]
# slopes              = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
# gradient_penalty    = tf.reduce_mean((slopes-1.)**2.)

########### Loss calculations #########################
kl_loss             = tf.reduce_mean(-sigmas +.5*(-1.+tf.exp(2.*sigmas)+tf.square(means)))  
recon_loss          = tf.reduce_mean(tf.square(real_models-G_dec))/2.
# computing for loss DCGAN

# d_real_loss = tf.reduce_mean(tl.cost.sigmoid_cross_entropy( tf.ones_like(D_legit), D_legit, name="loss_d_real"))
# d_fake_loss = tf.reduce_mean(tl.cost.sigmoid_cross_entropy( tf.zeros_like(D_fake),  D_fake,  name="loss_d_fake"))
# g_loss = tf.reduce_mean(tl.cost.sigmoid_cross_entropy( tf.ones_like(D_fake), D_fake, name="g_loss")) + (100)*recon_loss
# d_loss = d_fake_loss + d_real_loss + 10.*gradient_penalty

logits_diff_real_fake = D_legit - tf.reduce_mean(D_fake, axis=0, keepdims=True)
logits_diff_fake_real = D_fake - tf.reduce_mean(D_legit, axis=0, keepdims=True)

d_loss_real = tf.reduce_mean(tf.square(logits_diff_real_fake-1.0))
d_loss_fake = tf.reduce_mean(tf.square(logits_diff_fake_real+1.0))
d_loss = d_loss_real + d_loss_fake

g_loss_real = tf.reduce_mean(tf.square(logits_diff_fake_real-1.0))
g_loss_fake = tf.reduce_mean(tf.square(logits_diff_real_fake+1.0))
g_loss = g_loss_real + g_loss_fake

# d_loss = -tf.reduce_mean(D_legit) + tf.reduce_mean(D_fake) + 10.*gradient_penalty
# g_loss = -tf.reduce_mean(D_fake) + (100)*recon_loss
# d_loss = -tf.reduce_mean(tf.log(D_legit) + tf.log(1. - D_fake))
# g_loss = -tf.reduce_mean(tf.log(D_fake))
v_loss              = kl_loss + recon_loss 

############ Optimization #############
v_vars = tl.layers.get_variables_with_name('vae', True, True)
g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)

net_g.print_params(False)
net_d.print_params(False)
net_m.print_params(False)
net_s.print_params(False)

d_optim = tf.train.AdamOptimizer( learning_rate = 1e-4, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer( learning_rate = 1e-4, beta1=0.5).minimize(g_loss, var_list=g_vars)
v_optim = tf.train.AdamOptimizer( learning_rate = 1e-4, beta1=0.5).minimize(v_loss, var_list=v_vars)

####### Training ################
config = tf.ConfigProto(device_count={'GPU': 1})
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
tl.utils.set_gpu_fraction(gpu_fraction=0.998)
sess.run(tf.global_variables_initializer())

# load checkpoints
if args.load: 
    load_networks(checkpoint_dir, sess, net_g, net_d, net_m= net_m, net_s= net_s, epoch = args.load_epoch)
    #these keep track of the discriminaotrs loss, the reconstruction loss, and the reconstruction loss on a validatino set s
    if len(args.load_epoch)>1: 
        track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, iter_counter  = [],[],[],[],[],[],0
    else: 
        track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, iter_counter = load_values(save_dir, recon = True ,valid = True )
else:     
    track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, track_g_loss, track_g_loss_iter, iter_counter  = [],[],[],[],[],[],[],[],0
  

if len(args.load_epoch)>1: 
    start = int(args.load_epoch)
else: 
    start = 0 
iter_counter = iter_counter - (iter_counter %5)

if  args.train:
    files,valid = grab_files_images(args.images, args.data)
    valid_models, valid_images, _ = make_inputs_and_images(valid, args.data)

    logging.debug(len(files))
    Train_Dis = True
    for epoch in range(start, args.epochs):
        random.shuffle(files)

        # feed_dict = {}
        # drop_d_dict         = tl.utils.dict_to_one( net_d.all_drop )
        # drop_d2_dict        = tl.utils.dict_to_one( net_d2.all_drop )
        # drop_fake_d2_dict   = tl.utils.dict_to_one( net_fake_d2.all_drop )
        # drop_eval           = tl.utils.dict_to_one( net_eval.all_drop )
        
        # drop_g              = tl.utils.dict_to_one( net_g.all_drop )
        # drop_g2             = tl.utils.dict_to_one( net_g2.all_drop )
        #feed_dict.update(drop_d_dict)
        #feed_dict.update(drop_d2_dict)
        #feed_dict.update(drop_fake_d2_dict)
        #feed_dict.update(drop_eval)

        for idx in xrange(0, len(files)/args.batchsize):
            file_batch = files[idx*args.batchsize:(idx+1)*args.batchsize]
            models, batch_images, start_time = make_inputs_and_images(file_batch, args.data)
            
            # feed_dict.update(drop_d_dict)
            # feed_dict.update(drop_d2_dict)
            # feed_dict.update(drop_fake_d2_dict)
            # feed_dict.update(drop_eval)
            # feed_dict.update(drop_g)
            # feed_dict.update(drop_g2)
            # feed_dict = {images: batch_images, real_models:models}
            # feed_dict[images] = batch_images
            # feed_dict[real_models] = models
            #training the discriminator and the VAE's encoder 

            # if Train_Dis:
            # else:
                # ones = sess.run([D_eval], feed_dict={real_models: models})
            errD,_,errV,_,r_loss = sess.run([d_loss, d_optim, v_loss, v_optim, recon_loss] ,feed_dict={images: batch_images, real_models:models})            
            track_d_loss.append(-errD)
            track_d_loss_iter.append(iter_counter)
                        
            #training the gen / decoder and the encoder 
            # if iter_counter % 5 ==0:
            for i in range(2):
                errG,_,errV,_,r_loss= sess.run([g_loss, g_optim, v_loss, v_optim, recon_loss], feed_dict={images: batch_images, real_models:models})

            track_recon_loss.append(r_loss)
            track_recon_loss_iter.append(iter_counter)
            track_g_loss.append(errG)
            track_g_loss_iter.append(iter_counter)
            # Train_Dis = (cal_acc(zeros,ones)<0.95)# only train discriminator at certain level of accuracy 

            logging.debug("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f, v_loss: %.4f, r_loss: %.4f" % (epoch, args.epochs, idx, len(files)/args.batchsize, time.time() - start_time, errD, errG, errV, r_loss))           
            iter_counter += 1
            sys.stdout.flush()

        # feed_dict.update(drop_g)
        # feed_dict.update(drop_g2)

        #saving the model 
        if np.mod(epoch, args.save) == 0:
            save_networks(checkpoint_dir,sess, net_g, net_d, epoch, net_m,net_s)
        #saving generated objects
        if np.mod(epoch, args.sample) == 0:
            # del feed_dict[real_models]
            # feed_dict[images] = batch_images

            models,recon_models = sess.run([net_g2.outputs,net_g.outputs], feed_dict={images: batch_images})       
            save_voxels(save_dir, models, epoch, recon_models )
        #saving learning info 
        if np.mod(epoch, args.graph) == 0: 
            # feed_dict[real_models] = models
            # feed_dict[images] = batch_images
            r_loss = sess.run([recon_loss], feed_dict={images: batch_images, real_models:models})
            track_valid_loss.append(r_loss[0])
            track_valid_loss_iter.append(iter_counter)
            render_graphs(save_dir,epoch, track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss) #this will only work after a 50 iterations to allows for proper averating 
            save_values(save_dir,track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, track_g_loss, track_g_loss_iter) # same here but for 300 
else:
    # TODO WORK on testing here
    print("Testing")
    
