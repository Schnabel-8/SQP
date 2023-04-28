import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, hessian

from scipy.optimize import minimize, NonlinearConstraint

import numpy as np


def genW(hess_f_W,hess_ineq_W,x0_W):
        len_W=len(hess_ineq_W(x0_W))
        return lambda x,lmd:hess_f_W(x)-np.sum(np.reshape(lmd,(len_W,1,1))*np.array(hess_ineq_W(x0_W)),axis=0)
# d is a row vector
# this function compute the objective of subproblem
# for given W_k and g_k
# obj must use jnp instead of np
def obj(Wk,gk):
    return lambda d:0.5*jnp.dot(d,jnp.dot(Wk,d.T))+jnp.dot(gk,d)
# L1 penal function
def P(f_P,sigma_P,ineq_P):
    return lambda x:f_P(x)+sigma_P*(np.sum(np.abs(np.minimum(ineq_P(x),0,out=None))))
# gen_cons must use jnp instead of np
def gen_cons(cons_,cons_jac_,xk_):
    return lambda d:jnp.array(cons_(xk_))+jnp.dot(cons_jac_(xk_),d)

def sqp(f,ineq,x0,epsi=1e-3,sigma=0.5,rho=0.8):
    jac_f=lambda x:np.array(jacfwd(f)(x))
    hess_f=lambda x:np.array(hessian(f)(x))
    jac_ineq=lambda x:np.array(jacfwd(ineq)(x))
    hess_ineq=lambda x:np.array(hessian(ineq)(x))

    xk=x0
    lmdk=np.zeros(len(ineq(x0)))
    # caculate Wk
    Wker=genW(hess_f,hess_ineq,x0)
    # caculate gk
    gker=lambda x: np.array(jacfwd(f)(x))
    
    alpha_step=0.01
    alpha=np.arange(0,rho,alpha_step)
    plist=alpha
    
    iter=0
    
    while(True):
        Wk=Wker(xk,lmdk)
        gk=gker(xk)
        cons=gen_cons(ineq,jac_ineq,xk)
        len_cons=len(cons(x0))
        consjac=lambda x:np.array(jacfwd(cons)(x))
        conshess=lambda x,v:np.sum(np.reshape(np.array(v),(len_cons,1,1))*np.array(hessian(cons)(x)),axis=0)
        nonlinear_constraint = NonlinearConstraint(lambda x:np.array(cons(x)), 0, np.inf, jac=consjac, hess=conshess)
        obj_=obj(Wk,gk)
        objjac=lambda x:np.array(jacfwd(obj_)(x))
        objhess=lambda x:np.array(hessian(obj_)(x))
        res=minimize(obj_,x0,method='trust-constr',jac=objjac,hess=objhess,
                    constraints=[nonlinear_constraint],options={'verbose': 0})
        lmdk=np.reshape(np.array(res.v),-1)
        #print(np.reshape(lmdk,-1)
        dk=np.array(res.x)
        if(np.linalg.norm(dk,ord=2)<=epsi):
            break;
            
        pfunc=P(f,sigma,cons)
        count=0
        for alphak in alpha:
            plist[count]=pfunc(xk+alphak*dk)
            count+=1
        alphak=alpha_step*np.argmin(plist)
        xk=xk+alphak*dk
        iter+=1
        print("iter :  ",iter,"obj:   ",f(xk),"p:  ",pfunc(xk),"xk:  ",xk,"dk_norm:  ",np.linalg.norm(dk,ord=2),"alphak:   ",alphak)
    return xk
