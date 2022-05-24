import sys
import os
import math
import pickle
import torch
import numpy as np
from sisyphe.models import Vicsek
from sisyphe.kernels import lazy_interaction_kernel, lazy_quadratic, lazy_xy_matrix
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
import matplotlib
from pykeops.torch import LazyTensor

################################## MAIN CLASS #########################################

class Swarmalator(Vicsek):
    """Main class as a subclass of the Vicsek class. 

    Exclusive attributes:
        phase ((N,) Tensor): phases of the particles 
        nu_phase (float): Drift intensity of the phase alignment force. 
        sigma_phase (float): Noise intensity in the phase equation. 
        phase_strength (float): Intensity of the phase attraction-repulsion force.
        exterior_potential (fun): Compute the additional force exerted by the external potential.
        exterior_potential_param (dict): Parameters of the external potential. 
        phase_drift: phase attraction-repulsion force. 
        
    """

    def __init__(self, pos, vel, phase,
                 v,
                 sigma, nu,
                 interaction_radius,
                 box_size,
                 phase_strength,
                 nu_phase, sigma_phase,
                 exterior_potential=None,
                 exterior_potential_param=None,
                 boundary_conditions='open',
                 dt=.01,
                 block_sparse_reduction=False,
                 number_of_cells=1600):
        """_summary_

        Args:
            pos ((N,d) Tensor): Positions
            vel ((N,d) Tensor): Velocities
            phase ((N,) Tensor): Phases
            v (float): Speed. 
            sigma (_type_): Drift intensity of the velocity alignment force.
            nu (_type_): Noise intensity in the velocity alignment equation.
            interaction_radius (float): Interaction radius. 
            box_size float or list of size d): Box size.
            phase_strength (float): Intensity of the phase attraction-repulsion force 
            nu_phase (_type_): Drift intensity of the phase alignment force.
            sigma_phase (_type_): Noise intensity in the phase alignment equation.
            exterior_potential (fun, optional): Compute the additional force exerted by the external potential. Default is None.
             exterior_potential_param (dict, optional): Parameters of the external potential. Default is None. 
            boundary_conditions (list or str, optional): Boundary conditions.
            dt (float, optional): Time step. Default is 0.01.
            block_sparse_reduction (bool, optional): Use block sparse reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if **block_sparse_reduction** is True. Will be rounded to the nearest lower power :attr:`d`. Default is 1024.
        """
        super().__init__(pos=pos, vel=vel, v=v, sigma=sigma, nu=nu, interaction_radius=interaction_radius,
                         box_size=box_size, vision_angle=2*math.pi, axis=None,
                         boundary_conditions=boundary_conditions,
                         variant=None, options=None, numerical_scheme='projection', dt=dt,
                         block_sparse_reduction=block_sparse_reduction, number_of_cells=number_of_cells)
        self.phase = phase
        self.R = interaction_radius
        self.phase_strength = phase_strength
        self.nu_phase = nu_phase
        self.sigma_phase = sigma_phase
        self.exterior_potential = exterior_potential       #function 
        self.exterior_potential_param = exterior_potential_param       #dico
        self.name = 'Swarmalators'
        self.phase_drift = None
        self.add_target_option_method("apply_exterior_potential", self.apply_exterior_potential)
        self.parameters = 'N=' + str(self.N) \
                          + ' ; R=' + str(round(self.R, 2)) \
                          + ' ; nu=' + str(round(self.nu, 2)) \
                          + ' ; sigma=' + str(round(self.sigma, 2)) \
                          + ' ; v=' + str(round(self.v, 2)) \
                          + ' ; nu_phase=' + str(round(self.nu_phase, 2)) \
                          + ' ; sigma_phase=' + str(round(self.sigma_phase, 2)) \
                          + ' ; phase_strength=' + str(round(self.phase_strength, 2)) 

    
    
    def apply_exterior_potential(self, target):
        """Compute the force exerted by an external potential and add it to the velocity target. 

        Args:
            target ((N,d) Tensor): The target velocities for the N particles.  

        Returns:
            (N,d) Tensor: The sum of the velocity targets and the force exerted by the external potential. 
        """
        if self.exterior_potential is None:
            return target
        else:
            F = self.exterior_potential(self.pos,**self.exterior_potential_param)
            return target + F/self.sigma
        
        
    
    def phase_vel_targets(self, who=None, with_who=None, kernel=lazy_interaction_kernel, isaverage=True,
                                         **kwargs):
        r"""Return a Tensor of size (N,d+d+2) where
        
            - the d first columns contain the phase attraction-repulsion force for each particle ;
            - the d+1 to d+d columns contain the velocity target for each particle ;
            - the d+d+1 column contains the cos of the phase target for each particle ;
            - the d+d+2 column contains the sin of the phase target for each particle.
        """
        if who is None: who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        if with_who is None: with_who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        M = who.sum()
        xpos = self.pos[who, :]
        xvel = self.vel[who, :]
        phase_x = self.phase[who]
        N = with_who.sum()
        ypos = self.pos[with_who, :]
        yvel = self.vel[with_who, :]
        phase_y = self.phase[with_who]

        x = torch.cat((xpos,phase_x.reshape((M,1)),xvel),axis=1)
        y = torch.cat((ypos,phase_y.reshape((N,1)),yvel),axis=1)
        
        def binary_formula(x,y):
            xpos = x[:,0:self.d].detach().clone()
            ypos = y[:,0:self.d].detach().clone()
            
            ##### Linear potential #####
            normalization = math.pi / 3
            r = lazy_xy_matrix(xpos, ypos, self.L, self.bc)
            sq_dist = (r ** 2).sum(-1)
            sq_dist = sq_dist + (-sq_dist.sign() + 1.)  # case norm=0
            dist = sq_dist.sqrt()
            grad = (1. / normalization) * r/dist
            ############################################################
            
            ##### Quadratic potential #####
#             normalization = math.pi * (1/8 - 1/3 + 1/4)
#             grad = - (1. / normalization) * lazy_quadratic(x=xpos, y=ypos, R=self.R, L=self.L, boundary_conditions=self.bc)
            #############################################################
            
            
            ##### Phase attraction-repulsion #####
            phase_x = x[:,self.d].detach().clone()
            phase_y = y[:,self.d].detach().clone()
            phi_i = LazyTensor(phase_x.reshape((x.shape[0], 1))[:, None])
            phi_j = LazyTensor(phase_y.reshape((y.shape[0], 1))[None, :])
            sin_ij = (phi_j - phi_i).sin()
            phase_potential_ij = -grad*sin_ij
            # This is a (M,N,d) Lazy Tensor 
            #############################################################
            
            one_M = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
            lazy_one_M = LazyTensor(one_M[:,None])
            
            ##### Velocity target #####
            yvel = y[:,(self.d+1):].detach().clone()            
            lazy_yvel = LazyTensor(yvel[None,:,:])           
            lazy_velmatrix = lazy_one_M * lazy_yvel
            # This is a (M,N,d) Lazy Tensor
            #############################################################
            
            ##### cos and sin targets #####
            cos_y = phi_j.cos()
            sin_y = phi_j.sin()
            lazy_cosmatrix = lazy_one_M * cos_y
            lazy_sinmatrix = lazy_one_M * sin_y
            # These are two (M,N,1) Lazy Tensors
            #############################################################
            
            ##### Concatenation ####
            lazy_all = phase_potential_ij.concat(lazy_velmatrix.concat(lazy_cosmatrix.concat(lazy_sinmatrix)))            
            # This is a (M,N,d+d+2) Lazy Tensor
            #############################################################
            
            return lazy_all

        return self.nonlinear_local_average(binary_formula, x, y, who=who, with_who=with_who, isaverage=isaverage,
                                            kernel=kernel)
    
    def update(self):

        targets = self.phase_vel_targets()
        
        
        ##### Compute the phase attraction-repulsion force (multiply the target by a scaling factor and the force intensity). 
        
        phase_scale = 1. / (self.R ** (self.d + 1))
        self.phase_drift = self.phase_strength * phase_scale * targets[:,:self.d]
        
        ##### Compute the velocity targets. 
        
        nonnorm_velocity_target = targets[:,self.d:-2]
        norm_target = torch.norm(nonnorm_velocity_target,dim=1)
        bad_guys = norm_target == 0.
        if bad_guys.sum()>0:
            nonnorm_velocity_target[bad_guys,:] = nonnorm_velocity_target.new(bad_guys.sum(), self.d).normal_()
            print("I killed" + str(bad_guys.sum()) + "NaN.")
        velocity_target = self.kappa * nonnorm_velocity_target / torch.norm(nonnorm_velocity_target,dim=1).reshape((self.N,1))
        velocity_target = self.apply_exterior_potential(velocity_target)
        
        ##### Compute the phase targets. 
        
        mean_cos = targets[:,-2].reshape(self.N)
        mean_sin = targets[:,-1].reshape(self.N)
        phi_target = torch.remainder(torch.atan2(mean_sin, mean_cos), 2 * math.pi)
        
        ### Scheme ###
        self.pos += self.v * self.vel * self.dt + self.phase_drift * self.dt
        self.check_boundary()
        self.vel = self.one_step_velocity_scheme(velocity_target)
        self.phase += -self.nu_phase * torch.sin(self.phase - phi_target) * self.dt \
                      + math.sqrt(2 * self.sigma_phase * self.dt) * torch.randn(self.N, device=self.phase.device)
        self.phase = torch.remainder(self.phase, 2 * math.pi)
        info = {"position": self.pos, "velocity": self.vel, "phase": self.phase}
        return info

    def one_step_velocity_scheme(self, targets):
        r"""One time step of the numerical scheme for the velocity.
        Args:
            targets ((N,d) Tensor)
        Returns:
            (N,d) Tensor
        """
        dB = self.vel.new(self.N, self.d).normal_()
        inRd = self.sigma * targets * self.dt \
                + math.sqrt(2 * self.sigma * self.dt) * dB
        orth = torch.einsum('ij,ij->i', self.vel, inRd)
        proj = inRd - orth.reshape(self.N, 1) * self.vel
        # Stratonovich correction
        dv = proj - self.sigma * (self.d - 1) * self.vel * self.dt
        new_vel = self.vel + dv
        new_vel = new_vel / torch.norm(new_vel, dim=1).reshape((self.N, 1))
        return new_vel
    
    def global_order(self):
        r"""The global order parameter is defined by 
        
        .. math::
            \frac{1}{2N(N-1)}\sum_{i\ne j} (1+\cos(\varphi_i - \varphi_j))
        """
        phase_i = LazyTensor(self.phase[:,None,None])
        phase_j = LazyTensor(self.phase[None,:,None])
        phase_ij = phase_i - phase_j
        cos_ij = phase_ij.cos()
        order_ij = 0.5*(1. + cos_ij)
        order_i = 1./(self.N - 1.) * (order_ij.sum(1).reshape(self.N) - 1)
        return (1./self.N) * order_i.sum().item()
        
    
###########################################################################################



######################### PLOT AND SAVE ON THE GO ##############################################


def scatter_save(mechanism, time, N_displayed=None, order=True, show=True, save=False, path='simu'):
    r"""Scatter plots of the particles.
    
    Args:
        mechanism (Swarmalator)
        time (list): List of times (float) to save.
        N_displayed (int, optional): Number of displayed particles. Default is mechanism.N
        show (bool, optional): Display all the plots. Default is True
        save (bool, optional): Save the plots. Default is False 
        path (str, optional): The name of the folder where the data is saved. Default is "simu"
    """
    t = 0
    tmax = len(time) - 1
    percent = 0
    real_time = [0.]
    op = [mechanism.order_parameter]
    if N_displayed is None:
        N_displayed = mechanism.N
    if save:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, path, r'frames')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
    for it, info in enumerate(mechanism):
        real_time.append(real_time[-1] + mechanism.dt)
        if order:
            op.append(mechanism.order_parameter)
        if real_time[-1] / time[tmax] >= percent / 100:
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if abs(real_time[-1] >= time[t]):
            f = plt.figure(t, figsize=(6, 6))
            ax = f.add_subplot(111)

            x = mechanism.pos[:N_displayed, 0].cpu()
            y = mechanism.pos[:N_displayed, 1].cpu()
            phi = mechanism.phase[:N_displayed].cpu().numpy() / (2 * math.pi)
            size = 2 * mechanism.R / 5

            edgecolor = None
            color = phi

            ax.scatter(x, y, c=color, cmap='hsv', s=.2)

            ax.axis('equal')  # set aspect ratio to equal
            ax.axis([0, mechanism.L[0].cpu(), 0, mechanism.L[1].cpu()])
            ax.set_title(mechanism.name + '\n Parameters: ' + mechanism.parameters
                         + '\n Time=' + str(round(real_time[-1], 1)), fontsize=10)
            if save:
                f.savefig(f"{final_directory}/" + str(t) + ".png")
            if show:
                plt.show()
            else:
                plt.close()
            t += 1
        if t > tmax:
            break
    if order:
        x = np.array(real_time)
        y = np.array(op)
        op_plot = plt.figure(t, figsize=(6, 6))
        plt.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('order parameter')
        plt.axis([0, time[tmax], 0, 1])
        if save:
            simu_dir = os.path.join(current_directory, path)
            op_plot.savefig(f"{simu_dir}/order_parameter.png")
        return x, y
    
#################################################################################################


def heatmap_gridplot_save(mechanism, time, R=None, K=None, order=True, show=True, save=False, save_order=True, path='simu'):
    r"""Plot the heatmap of the density of particles, the average phase on a uniform grid, the average velocity field and the position of three individual particles.  
    
    Args:
        mechanism (Swarmalator)
        time (list): List of times (float) to save.
        R (float, optional): Radius of the displayed particles. Default is 5 * mechanism.R
        K (int, optional): Number of cells per dimension. Default is floor(1/mechanism.R)
        order (bool, optional): Compute the velocity order parameter. Default is False
        show (bool, optional): Display all the plots. Default is True
        save (bool, optional): Save the plots. Default is False 
        save_order (bool, optional): Compute and save the glbal phase order parameter. Default is True
        path (str, optional): The name of the folder where the data is saved. Default is "simu"
    """
    t = 0
    tmax = len(time) - 1
    percent = 0
    real_time = [time[0]]
    op = [mechanism.order_parameter]
    if R is None:
        R = 5 * mechanism.R
    if K is None:
        K = int(1./mechanism.R)
    if save:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, path, r'frames')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
    if save_order:
        data_file = open(os.path.join(current_directory, path, r'phase_order.pkl'), "wb")
        data = {"time" : [0.], "phase_order" : [mechanism.global_order()]}
        pickle.dump(data, data_file)
        data_file.close()
    for it, info in enumerate(mechanism):
        real_time.append(real_time[-1] + mechanism.dt)
        if order:
            op.append(mechanism.order_parameter)
        if (real_time[-1] - time[0]) / (time[tmax] - time[0]) >= percent / 100:
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if abs(real_time[-1] >= time[t]):
            index_x = torch.floor((mechanism.pos[:,0] / mechanism.L[0]) * K)
            index_y = torch.floor((mechanism.pos[:,1] / mechanism.L[1]) * K)

            phase_mean = np.zeros((K,K))
            for kx in range(K):
                for ky in range(K):
                    index = ((index_x == kx) & (index_y == ky))
                    cos_mean = torch.cos(mechanism.phase[index]).sum()
                    sin_mean = torch.sin(mechanism.phase[index]).sum()
                    phase_mean[kx,ky] = torch.remainder(torch.atan2(sin_mean,cos_mean),2*math.pi).item()/(2*math.pi)
            
            Kvel = 20
            r = float(mechanism.L[0].cpu())/Kvel
            veltheta_mean = []
            index_x = torch.floor((mechanism.pos[:,0] / mechanism.L[0]) * Kvel)
            index_y = torch.floor((mechanism.pos[:,1] / mechanism.L[1]) * Kvel)
            for kx in range(Kvel):
                for ky in range(Kvel):
                    index = ((index_x == kx) & (index_y == ky))
                    vel_mean = mechanism.vel[index,:].sum(0)
                    veltheta_mean.append(torch.remainder(torch.atan2(vel_mean[1],vel_mean[0]),2*math.pi).item())
            veltheta_mean = np.array(veltheta_mean)

            K2 = (mechanism.L.cpu() / r).floor().int().type(torch.LongTensor)
            vx = torch.tensor(range(K2[0]))
            vy = torch.tensor(range(K2[1]))
            x = vx.repeat(K2[1:].prod())
            x = x.reshape(K2.prod(), 1)
            y = vy.repeat_interleave(K2[0])
            y = y.reshape(K2.prod(), 1)
            centroids = r * torch.cat((x, y), dim=1) + r / 2


            f = plt.figure(t, figsize=(14, 6))
            
            ############### Heatmap of the density of particles ###############
            
            axh = f.add_subplot(121)
            plt.set_cmap("hot")
            L0 = float(mechanism.L[0].cpu())
            L1 = float(mechanism.L[1].cpu())
            x = mechanism.pos[:, 0].cpu().numpy()
            y = mechanism.pos[:, 1].cpu().numpy()
            counts, xedges, yedges, im = axh.hist2d(x, y, bins=K, range=np.array([[0, L0], [0, L0]]), density=True)
            axh.quiver(centroids[:,0].numpy(),centroids[:,1].numpy(),np.cos(veltheta_mean),np.sin(veltheta_mean))
            axh.axis('equal')  # set aspect ratio to equal
            axh.axis([0, L0, 0, L1])
            axh.set_title(mechanism.name + '\n Time=' + str(round(real_time[-1], 1)), fontsize=10)
            im.set_clim(0, 4)
            cb = f.colorbar(im, ax=axh)
            
            ###################################################################
            
            ################# Average phase on a uniform grid #################
            
            ax = f.add_subplot(122)

            phase_plot = ax.imshow(np.transpose(phase_mean), vmin=0, vmax=1, cmap='hsv',
                      aspect='equal',origin="lower",interpolation='nearest')

            x = np.array([mechanism.pos[42, 0].item()*K, mechanism.pos[0, 0].item()*K, mechanism.pos[33, 0].item()*K])
            y = np.array([mechanism.pos[42, 1].item()*K, mechanism.pos[0, 1].item()*K, mechanism.pos[33, 1].item()*K])
            size = [R * K, R * K, R * K]
            offsets = list(zip(x, y))
            color1 = mechanism.phase[42].item()/(2*math.pi)
            color2 = mechanism.phase[0].item()/(2*math.pi)
            color3 = mechanism.phase[33].item()/(2*math.pi)
            hsv_colormap = matplotlib.cm.get_cmap('hsv')
            rgb_tuple = [hsv_colormap(color1)[:-1],hsv_colormap(color2)[:-1],hsv_colormap(color3)[:-1]]
            ax.add_collection(EllipseCollection(
                widths=size, heights=size, angles=0, units='xy', cmap='hsv', facecolor=rgb_tuple,
                edgecolor='k', linewidth=1.5, offsets=offsets, transOffset=ax.transData))
            ax.quiver(K/L0 * centroids[:,0].numpy(),K/L0 * centroids[:,1].numpy(),np.cos(veltheta_mean),np.sin(veltheta_mean))
            cb = f.colorbar(phase_plot, ax=ax)
            
            ax.set_title(mechanism.name + '\n Time=' + str(round(real_time[-1], 1)), fontsize=10)
            ax.set(xlim=[0,K],
                   xticks=[0, K/5., 2*K/5, 3*K/5, 4*K/5, K],
                   xticklabels=['0', '0.2', '0.4', '0.6', '0.8', '1'],
                   ylim=[0,K],
                   yticks=[0, K/5., 2*K/5, 3*K/5, 4*K/5, K],
                   yticklabels=['0', '0.2', '0.4', '0.6', '0.8', '1'])
            
            ###################################################################
            
            ############################## Save ###############################
            
            if save:
                f.savefig(f"{final_directory}/" + str(t) + ".png")
            if show:
                plt.show()
            else:
                plt.close()
            if save_order:
                data["time"].append(real_time[-1])
                data["phase_order"].append(mechanism.global_order())
                data_file = open(os.path.join(current_directory, path, r'phase_order.pkl'), "wb")
                pickle.dump(data, data_file)
                data_file.close()
            t += 1
        if t > tmax:
            break
    if order:
        x = np.array(real_time)
        y = np.array(op)
        op_plot = plt.figure(t, figsize=(6, 6))
        plt.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('order parameter')
        plt.axis([0, time[tmax], 0, 1])
        if save:
            simu_dir = os.path.join(current_directory, path)
            op_plot.savefig(f"{simu_dir}/order_parameter.png")
        return x, y

