import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, hessian

from scipy.optimize import minimize, NonlinearConstraint

import numpy as np

import json as js

import matplotlib as mpl
import matplotlib.pyplot as plt

import os


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
    data=[list(x0)]
    hashcode=hex(hash(hash(f)+hash(ineq)))
    
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
        data.append(list(xk))
        print("iter :  ",iter,"obj:   ",f(xk),"p:  ",pfunc(xk),"xk:  ",xk,"dk_norm:  ",np.linalg.norm(dk,ord=2),"alphak:   ",alphak)
    
    if not os.path.exists("cache"):
        os.mkdir("cache")
    with open('./cache/'+str(hashcode)+'.json', 'w') as f:
        js.dump(data, f)
    return [xk,hashcode]


def myplot(f,hashcode,xl=0,xr=1,yl=0,yr=1):
    fd=open("./cache/"+str(hashcode)+".json")
    data=js.load(fd)
    len_d=len(data)
    xd=[]
    yd=[]
    for i in range(0,len_d):
        xd.append(data[i][0])
        yd.append(data[i][1])
    
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 

    stepx = 0.01; stepy = 0.01 

    x = np.arange(xl,xr,stepx); y = np.arange(yl,yr,stepy) 

    X,Y = np.meshgrid(x,y) 

    Z = f(X, Y)
    fig, ax = plt.subplots(figsize=(8,8),dpi=100)

    CS = ax.contourf(X, Y, Z,cmap=mpl.cm.rainbow)
    plt.colorbar(CS)

    CS = ax.contour(X, Y, Z)

    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    
    plt.xticks(np.arange(0,1,0.1))
    plt.yticks(np.arange(0,1,0.1))
    
    plt.plot(xd,yd,color='yellow',marker='o', markerfacecolor='white')
    
    plt.tight_layout()
    plt.show()