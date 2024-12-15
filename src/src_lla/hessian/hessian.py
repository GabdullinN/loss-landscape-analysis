# Hessian decomposition class and its methods (calculation of eigenvalues, spectral decomposition, etc) 

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024


import torch
import copy
import numpy as np

from src_lla.loss_landscapes.model_interface.model_wrapper import wrap_model
from src_lla.loss_landscapes.model_interface.model_parameters import rand_u_like, rand_n_like, orthogonal_to
from src_lla.hessian.utils import list_prod, update_vect, list_norm, get_params_grads, ortho_vect, hes_prod


def tol_check(a1,a2,a3,c,tol):
    return abs(a1 - a2) / (abs(a3) + c) < tol
    

class hessian_calc():
    def __init__(self, model, metric): 
        """
        creates computational graph for grad evaluation

        make sure to use .reset() on every hessian_calc object instance after use; failure to do so may lead to memory leak
        
        :model - torch model object
        :metric - loss landscape analysis metric object
        """

        # all input data is in metric object
        self.metric = metric
        self.device = metric.device
        
        self.model = model
        self.wrapped_model = wrap_model(copy.deepcopy(model))
        self.wrapped_parameters = self.wrapped_model.get_module_parameters()

        # creating computation graph for data in metric and getting grad info
        loss, outputs = self.metric(None,model=self.model,use_wrapper=False,return_pred=True)
        loss.backward(create_graph=True) # torch.autograd.grad(...) is recommended instead
        self.params, self.grads = get_params_grads(self.model)


    def reset(self):
        """
        sets grads to None to release old computational graph and allow its removal from memory
        operates in way similar to optim.zero_grad()
        """


        self.grads = None
        #self.params = None
        self.wrapped_parameters = None
        self.wrapped_model = None
        
        for layer, p in self.model.named_parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        

    def eigs_calc(self, n_iter=100, tol=1e-3, top_n=1):
        """
        estiamates top_n eigenvalues of the hessian using power iteration
        
        :n_iter - number of iterations
        :tol - tolerance to compare eigenvalues on consecutive iterations
        :top_n - number of top eigenvalues to compute
        """

        eigenvalues = []
        eigenvectors = []
        c = 1e-6 # for numerical stability

        for _ in range(top_n):

            eigenvalue = None
            v = rand_n_like(self.wrapped_parameters)
            v = list_norm(v)

            done = False
            for i in range(n_iter):
                if done:
                    break
                v = ortho_vect(v, eigenvectors)
                self.model.zero_grad()

                # calculating eigenvalues via hessian
                H = hes_prod(self.grads, self.params,v)
                tmp_eigenvalue = list_prod(H, v).cpu().item()
                v = list_norm(H)

                # check if tolerance condition is satisfied
                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    done = tol_check(eigenvalue,tmp_eigenvalue,eigenvalue,c,tol)
                    if not done:
                        eigenvalue = tmp_eigenvalue
                        
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

        return eigenvalues, eigenvectors


    def Lacsoz_step(self,step_num,vs,ws,alphas,betas):
        """
        step of SLQ for internal use by esd_calc
        """

        # take v from previous step
        v = vs[-1]
        
        if step_num != 0:
            # beta and v update
            w = ws[-1]
            beta = torch.sqrt(list_prod(w, w)) # norm of w from prev step
            betas.append(beta.cpu().item())
            if beta != 0.:
                v = ortho_vect(w, vs)
                vs.append(v)
            else:
                # generate a new vector
                w = rand_n_like(self.wrapped_parameters)
                v = ortho_vect(w, vs)
                vs.append(v)
    
        # w and alpha update
        self.model.zero_grad()
        w_prime = [torch.zeros(p.shape).to(self.device) for p in self.params]
        w_prime = hes_prod(self.grads, self.params,v)
        alpha = list_prod(w_prime, v)
        alphas.append(alpha.cpu().item())
        w = update_vect(w_prime, v, alpha=-alpha)
    
        if step_num != 0:
            # add vj-1 (step-2) contrib
            w = update_vect(w, vs[-2], alpha=-beta)
        
        ws.append(w)
        
        
    def Rademacher_vect(self):
        """
        Generate Rademacher random variables (a random vector of 0s and 1s with 0s converted to -1s)
        """
        v = [torch.randint_like(p, high=2).to(self.device) for p in self.params]
        for v_i in v:
            v_i[v_i == 0] = -1

        return v
    
    
    def tr_calc(self, n_iter=100, tol=1e-3):
        """
        Trace computation using Hutchinson's method
        
        :n_iter - maximum number of iterations 
        :tol - tolerance
        """

        trace_list = []
        prev_trace = np.zeros(0)
        c = 1e-6 # for numerical stability

        for i in range(n_iter):
            self.model.zero_grad()
            v = self.Rademacher_vect()

            H = hes_prod(self.grads, self.params,v)
            cur_trace = list_prod(H, v)
            trace_list.append(cur_trace.cpu().item())
            done = tol_check(np.mean(trace_list),prev_trace,prev_trace,c,tol)
            #if abs(np.mean(trace_list) - prev_trace) / (abs(prev_trace) + c) < tol:
            if done:
                return np.mean(trace_list)
            else:
                prev_trace = np.mean(trace_list)

        return np.mean(trace_list)
        

    def esd_calc(self, n_iter=100, n_v=1,max_v=10):
        """
        estimates eigenvalue spectral decomposition (esd) using stochastic lanczos quadrature (SLQ)
        
        :n_iter - number of iterations
        :n_v - number of SLQ runs
        :max_v - max number of vectors stored to create new orthogonal vects*
        *each vect has size of model weights, therefore memory requirements of one evaluation are model_size*(1+max_v). 
        *High max_v value may lead to memory issues
        """

        all_eigs = []
        all_weights = []

        for k in range(n_v):

            # Laczos algorithm initialization: generate the initial vector
            v = list_norm(self.Rademacher_vect())
        
            vs = [v]
            ws = []
            alphas = []
            betas = []
            
            # starting iterations
            for cur_step in range(n_iter):
                self.Lacsoz_step(cur_step,vs,ws,alphas,betas)
                    
                # removing extra vects due to memory considerations 
                if len(vs) > max_v+1:
                    idx = max_v+1
                    vs = vs[-idx:]

            # form T matrix from alphas and betas
            T = torch.zeros(n_iter, n_iter).to(self.device)
            for i in range(len(alphas)):
                T[i, i] = alphas[i]
                if i < len(alphas) - 1:
                    T[i + 1, i] = betas[i]
                    T[i, i + 1] = betas[i]

            # compute eigenvalue, eigenvector pairs
            eigenvalues, eigenvectors = torch.linalg.eigh(T)

            eigs = eigenvalues.real
            weights = torch.pow(eigenvectors[0,:], 2)
            all_eigs.append(list(eigs.cpu().numpy()))
            all_weights.append(list(weights.cpu().numpy()))

        return all_eigs, all_weights
