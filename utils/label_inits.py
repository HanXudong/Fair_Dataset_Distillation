#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:44:57 2019

@author: ilia10000
"""
import numpy as np
# from skimage.measure import compare_mse, compare_nrmse, compare_ssim
from utils.baselines import average_train
import torch

def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]
def rvs(dim):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
 
def load_embeddings():
    embeddings_dict = {}
    with open("glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict
    
def distillation_label_distance_based_initialiser(state, distance_matrix):
    num_classes=state.num_classes
    if state.num_classes==2:
        dl_array = [[i==j for i in range(1)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
    else:
        dl_array = [[i==j for i in range(num_classes)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
    new_array = np.array(dl_array, dtype=float)
    
    #move the vectors closer based on distances
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i!=j:
                new_array[i]=np.add(new_array[i],np.multiply(dl_array[j],(1-distance_matrix[i][j])))
    return new_array
    
        
def images_dist(dist_metric, reverse, images):
   try:
       imgs = np.array(images)
   except:
       imgs = np.array(images.cpu())
   if len(imgs.shape)==4:
       imgs = np.moveaxis(imgs, 1, -1)       
   dist_mat = np.zeros((len(imgs),len(imgs)))
   for i in range(len(imgs)):
       for j in range(len(imgs)):
           if dist_metric=="MSE":
               dist_mat[i,j] = compare_mse(imgs[i], imgs[j])
           elif dist_metric=="NRMSE":
               dist_mat[i,j] = compare_nrmse(imgs[i], imgs[j])
           elif dist_metric=="SSIM":
               dist_mat[i,j] = 1-compare_ssim(imgs[i], imgs[j], win_size=3, multichannel=True)
           
   dist_mat = np.divide(dist_mat, np.max(dist_mat))
   if reverse:
       dist_mat = np.subtract(1, dist_mat)
   return dist_mat

def distillation_label_initialiser(state, num_per_step, dtype, req_lbl_grad):
    init_type=state.random_init_labels
    device=state.device
    num_classes=state.num_classes
    label_smoothing=0.1
    if init_type=="stdnormal":
        if state.num_classes == 2:
            dl_array = np.random.normal(size=(num_per_step, 1))
            #dl_array = torch.randn(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.random.normal(size=(num_per_step, num_classes))
            #dl_array = torch.randn(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
    elif init_type=="uniform":
        if num_classes == 2:
            dl_array = np.random.uniform(size=(num_per_step, 1))
            #dl_array = torch.rand(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.random.uniform(size=(num_per_step, num_classes))
            #dl_array = torch.rand(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
    elif init_type=="bin":
        if num_classes == 2:
            dl_array = np.random.binomial(1,0.5,size=(num_per_step, 1))
            #dl_array = torch.rand(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.random.binomial(1,0.5,size=(num_per_step, num_classes))
            #dl_array = torch.rand(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad) 
    elif init_type=="zeros":
        if num_classes == 2:
            dl_array = np.zeros((num_per_step, 1))
        else:
            dl_array = np.zeros((num_per_step, num_classes))
    elif init_type=="ones":
        if num_classes == 2:
            dl_array = np.ones((num_per_step, 1))
            #distill_label = torch.ones(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.ones((num_per_step, num_classes))
            #distill_label = torch.ones(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
    elif init_type=="hard":
        if state.num_classes==2:
            dl_array = [[i==j for i in range(1)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        else:
            dl_array = [[i==j for i in range(num_classes)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="smoothed":
        if state.num_classes==2:
            dl_array = [[i==j for i in range(1)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        else:
            dl_array = [[i==j for i in range(num_classes)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        dl_array=np.add(np.multiply(dl_array,(1-label_smoothing)), label_smoothing/num_classes)
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="orthogonal":
        M = rvs(num_classes)
        #This means that if you have multiple images per class per step, all labels for same class are same
        dl_array = M*state.distilled_images_per_class_per_step 
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="file":
        with open("labels.txt") as f:
            dl_array = [[float(l) for l in line.strip().split(", ")] for line in f.readlines()]
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="CNDB":
        embed_dict = load_embeddings()
        embdeddings = [embed_dict[str(name)] for name in state.dataset_labels]
        distances= images_dist("MSE", state.invert_dist, embdeddings)
        dl_array=distillation_label_distance_based_initialiser(state,distances)
    elif init_type=="AIDB":
        avg_imgs = average_train(state)[0][0]
        distances= images_dist(state.dist_metric, state.invert_dist, avg_imgs)
        dl_array=distillation_label_distance_based_initialiser(state,distances)
    
    if state.add_first:
        dl_array=np.add(dl_array,state.add_label_scaling)
        dl_array=np.multiply(dl_array,state.mult_label_scaling)
    else:
        dl_array=np.multiply(dl_array,state.mult_label_scaling)
        dl_array=np.add(dl_array,state.add_label_scaling)
    distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    return distill_label