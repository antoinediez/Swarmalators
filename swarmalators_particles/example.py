import sys
import os
import cv2
import math
import time
import torch
import numpy as np
import scipy.integrate as integrate
from sisyphe.display import save
import swarmalators 

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


######################## PARAMETERS #################################################

N = 1000000    # Number of particles 

L = 1.   # Size of the domain
R = .01    # Interaction radius
c = 1.    # Speed   

# Velocity alignment parameters
nu = 5./R
sigma = 1./R
kappa = nu/sigma

# Phase alignment parameters
nu_phase = 3.0/R
sigma_phase = 1.0/R
kappa_phase = nu_phase/sigma_phase

phase_strength = 0.2    # Phase attraction-repulsion force intensity

# Choose the time-step sufficiently small
max_coef = max(nu,sigma,nu_phase,sigma_phase,abs(phase_strength/ (R ** 2)))
dt = .01/max_coef

simu_time = 15.    # Simulation time
start_time = 0.    # Initial time
simu_name = "simu" 

#####################################################################################

######################### C1 and C2 #################################################


def c1(k):
    integrandeZ = lambda t: np.exp(k * np.cos(t))
    integrande0 = lambda t: np.cos(t) * np.exp(k * np.cos(t))
    Z = integrate.quad(integrandeZ,0,math.pi)
    I0 = integrate.quad(integrande0,0,math.pi)
    return I0[0]/Z[0]

def c2p(k):
    def g(phi,k):
        integrande = lambda t: np.exp(-k * np.cos(t))
        I1 = integrate.quad(integrande,0,phi)
        I0 = integrate.quad(integrande,0,math.pi)
        return phi / k - math.pi / k * I1[0] / I0[0]
    integrande1 = lambda phi: np.cos(phi) * g(phi,k) * np.exp(k * np.cos(phi)) * np.sin(phi)
    integrande0 = lambda phi: g(phi,k) * np.exp(k * np.cos(phi)) * np.sin(phi)
    I1 = integrate.quad(integrande1,0,2*math.pi)
    I0 = integrate.quad(integrande0,0,2*math.pi)
    return I1[0]/I0[0]

c1_phase = c1(kappa_phase)
c2_phase = c2p(kappa_phase)

#####################################################################################

######################### SAMPLE INITIAL PARTICLES ##################################

# Positions
pos = L*torch.rand((N,2)).type(dtype)

# The individual positions of the following particles will be displayed
pos[0,:] = torch.tensor([L/2,.15]).type(dtype)  
pos[33,:] = torch.tensor([L/2,.5]).type(dtype)
pos[42,:] = torch.tensor([L/2,.85]).type(dtype)

# Velocities and phase 

### Uniform ###
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))
phase = 2 * math.pi * torch.rand(N).type(dtype)
###############

# ### Gradient ###
# thet = math.pi/2 + 0.4 * (torch.rand(N).type(dtype) - 0.5)
# u1 = torch.cos(thet).reshape((N,1))
# u2 = torch.sin(thet).reshape((N,1))
# vel = torch.cat((u1,u2),dim=1)
# pos_center = pos - torch.tensor([[L/2,L/2]]).type(dtype)
# phase = 2 * math.pi * pos_center[:,1]
# phase = torch.remainder(phase, 2*math.pi)
# ################


#####################################################################################


########################## CREATE A SYSTEM OF PARTICLES #############################

simu=swarmalators.Swarmalator(pos=pos,vel=vel,phase=phase,
             v=c,
             sigma=sigma,nu=nu,
             interaction_radius=R,
             phase_strength = phase_strength,
             nu_phase = nu_phase,
             sigma_phase = sigma_phase,
             box_size=L,
             dt=dt,
             boundary_conditions='periodic',
             block_sparse_reduction=True,
             number_of_cells=90**2)

#####################################################################################

print(simu.parameters)

##################### RUN THE SIMULATION AND SAVE ###################################

frames = np.arange(start_time,start_time + simu_time,0.01)

s = time.time()
swarmalators.heatmap_gridplot_save(simu, frames, R=None, K=None, order=False, show=False, save=True, path=simu_name)
e = time.time()
print("Simulation time: " + str(e-s) + " seconds")

torch.save(simu.pos, simu_name+"/final_pos.p")
torch.save(simu.vel, simu_name+"/final_vel.p")
torch.save(simu.phase, simu_name+"/final_phase.p")


#####################################################################################


##################### MAKE A VIDEO!  ################################################

def make_video(number_of_frames,directory,video_name='funny_video',rate=40):
    img_array = []
    os.chdir(directory)
    current_directory = os.getcwd()
    frame_directory = current_directory+'/frames/'
    for count in range(number_of_frames):
        filename = frame_directory+str(count)+'.png'
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(video_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), rate,size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    os.chdir('..')

make_video(len(frames),simu_name,video_name=simu_name)

