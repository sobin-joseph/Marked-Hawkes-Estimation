#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# In[28]:

dict1={}
support=20
epsilon = 1e-8
nNeurons=32
#Inflection Points for kernel
def inflectionPoints(A,B):
    dict1.clear()
    div = B[0]+epsilon*(np.abs(B[0])<epsilon)
    x = -B[1]/div
    interestX1 = x*(x>0)
   
    alwaysInclude = (x<=0)*(B[0]>0) #dont change
    alwaysExclude = (x<=0)*(B[0]<0)
    
    tempX = x*(~alwaysInclude)*(~alwaysExclude)
    interestX1 = tempX[interestX1>0]
    interestX = np.sort(interestX1)
    interestX = np.append(0,interestX)
    
    dict1['inflection']=interestX
    
    return

#combined Mu and Kernel
def nnMuKernel(x,diffx,mu,c,A,B):
    kernelSum=(nnKernel(diffx,A,B)).sum()
    y=max(mu+kernelSum,0)+c
    return y

#Kernel Calcualtion
def nnKernel(x,A,B):
    alphas = A[0]
    alpha0 = A[1]
    betas = B[0]
    beta0 = B[1]
    n1 = np.maximum(betas*x.reshape(1,-1) + beta0,0.)
    y = np.dot(alphas.T,n1) + alpha0

    return y

#integrated mu and kernel
def nnIntegrateMuKernel(iArray,mu,c,t,A,B,dict1):
    IntegratedLambda=np.array(Parallel(n_jobs=50)(delayed(nnIntegratedParallel)(j,mu,c,t,A,B,dict1) for j in np.nditer(iArray)),dtype=object)
    return IntegratedLambda.sum()

def nnIntegratedParallel(j,mu,c,t,A,B,dict1):
    inflectionKernel=dict1['inflection']
    tj=t[j]
    if tj>0:
        iP=t[j-1]
        lj=max(0,j-support)
        temp=t[lj:j]
        inflectionPs=inflectionKernel+temp.reshape(-1,1)
        inflectionPs=inflectionPs.reshape(-1)
        inflectionPs=np.sort(inflectionPs)
        inflectionPs=inflectionPs[(inflectionPs>iP)*(inflectionPs<tj)]
        inflectionPs=np.append(iP,inflectionPs)
        inflectionPs=np.append(inflectionPs,tj)
        IntegratedLambda=nnIntegrateMuKernelPart(tj,iP,temp,inflectionPs,mu,c,A,B)
    else:
        IntegratedLambda=0
    return IntegratedLambda
            
def outerIps(inflectionPs,temp,mu,A,B):
    alphas = A[0]
    alpha0 = A[1]
    betas = B[0]
    beta0 = B[1]
    inflection=[]
    infl1=-100
    for j in range(1,len(inflectionPs)):
        iP1=inflectionPs[j]
        iP2=inflectionPs[j-1]
        v1=-mu
        v2=0
        
        n1=betas*(iP1-temp-epsilon).reshape(1,-1)+beta0
        dn1=(n1>0)
        v3=-len(temp)*alpha0[0]-(alphas*beta0*dn1).sum()+(alphas*betas*dn1*(temp).reshape(1,-1)).sum()
        v4=(alphas*betas*dn1).sum()
        infl=(v1+v3)/(v2+v4)
        if (iP1>infl) & (iP2<infl) &(infl1!=infl):
            infl1=infl
            inflection.append(infl)
    return np.array(inflection)

def lambdaVs(temp,inflectionPs,mu,A,B):
    alphas = A[0]
    alpha0 = A[1]
    betas = B[0]
    beta0 = B[1]
    n1=np.maximum(betas*(inflectionPs.reshape(-1,1)-temp).reshape(1,-1)+beta0,0)
    n2=np.sum(A[0]*n1,axis=0)+A[1]
    n2=n2.reshape(len(inflectionPs),len(temp))
    y=np.sum(n2,axis=1)+mu
    return y
    
def nnIntegrateMuKernelPart(tj,iP,temp,inflectionPs,mu,c,A,B):
  
    alphas = A[0]
    alpha0 = A[1]
    betas = B[0]
    beta0 = B[1]

 
    lambdaValInfl=lambdaVs(temp,inflectionPs,mu,A,B)
    
    
    if len(lambdaValInfl<=0)!=0:
        oIp=[]
        oIp.append(inflectionPs[0])
        for j in range(1,len(inflectionPs)):
            if ((lambdaValInfl[j]<=0) and (lambdaValInfl[j-1]>0)) or ((lambdaValInfl[j]>0) and (lambdaValInfl[j-1]<=0)):
                if (oIp[-1]==inflectionPs[j-1]):
                    oIp.append(inflectionPs[j])
                else:
                    oIp.append(inflectionPs[j-1])
                    oIp.append(inflectionPs[j])
        oIp=np.array(oIp)
        outerInflections=outerIps(oIp,temp,mu,A,B)
        if len(outerInflections)!=0:
            inflectionPs=np.concatenate((inflectionPs,outerInflections))
            inflectionPs=np.sort(inflectionPs)
    
    integral=0
    for k in range(1,len(inflectionPs)):
        iP1=inflectionPs[k]
        iP2=inflectionPs[k-1]
        
        n1=np.maximum(betas*(iP1-temp-epsilon).reshape(1,-1)+beta0,0)
        dn1=(n1>0)
        n2=(np.dot(alphas.T,n1)+alpha0).sum()
        
        dn2=((mu+n2)>0)
        if dn2!=0:
            term1Mu=mu*iP1
            term2Mu=mu*iP2
            temp1=(iP1-temp).reshape(1,-1)
            temp2=(iP2-temp).reshape(1,-1)
        
            term1Kernel=(alpha0*temp1+np.sum(alphas*temp1*(beta0+0.5*temp1*betas)*dn1,axis=0)).sum()
            term2Kernel=(alpha0*temp2+np.sum(alphas*temp2*(beta0+0.5*temp2*betas)*dn1,axis=0)).sum()
        
        
            integral+=(term1Mu+ term1Kernel-(term2Mu+ term2Kernel))*dn2+c*(iP1-iP2)
     
    return integral
        

   
    
#Mu and Kernel Likelihood
def nnLoglikelihoodVal(iArray,t,mu,c,A,B,dict1):
    ll=0
    llIn=nnIntegrateMuKernel(iArray,mu,c,t,A,B,dict1)
    print("Integrated Part Mu and kernel ",llIn)
    ll=ll+llIn
    for i in range(1,len(t)):
        
        li = max(i-support,0)
        
        temp = t[i]-t[li:i]
        MuKernel=nnMuKernel(t[i],temp,mu,c,A,B)

        logLam = -np.log(MuKernel)
      
        ll = ll+logLam
    print("log Part",ll-llIn.sum())
    return ll    


#Mu and Kernel combined Gradients
def gradientsMuKernel(mu,c,train,iArray,A,B,dict1):
    A_grad=[]
    B_grad=[]
    mus1grad=[]
    musCgrad=[]
    
    A_grad.append(np.zeros([nNeurons,1]))
    A_grad.append(0)
    B_grad.append(np.zeros([nNeurons,1]))
    B_grad.append(np.zeros([nNeurons,1]))
    mus1grad.append(0)
    musCgrad.append(0)
    
    alphas = A[0]
    gradMu=0
    
    gradA=np.zeros((len(alphas),1))*0
    gradB=np.zeros((len(alphas),1))*0
    gradB0=np.zeros((len(alphas),1))*0
    gradA0=0
    gradC=0
    
    
    gradients=np.array(Parallel(n_jobs=50)(delayed(gradientsMuKernelParallel)(mu,c,train,j,A,B,dict1) for j in np.nditer(iArray)),dtype=object)
    gradMu+=(gradients[:,0]).sum()

        
    gradA+=(gradients[:,1]).sum()
    gradA0+=(gradients[:,2]).sum()
    gradB+=(gradients[:,3]).sum()
    gradB0+=(gradients[:,4]).sum()
    gradC+=(gradients[:,5]).sum()
    
    length=len(iArray)
        
    gradMu=gradMu/length
    gradA,gradA0,gradB,gradB0=gradA/length,gradA0/length,gradB/length,gradB0/length
  
    


    gradC  = gradC/length
    
    mus1grad[0]=gradMu
    A_grad[0] = gradA.astype('float32')
    A_grad[1] = gradA0
    B_grad[0] = gradB.astype('float32')
    B_grad[1] = gradB0.astype('float32')
    musCgrad[0]=gradC
    
    return A_grad,B_grad,mus1grad,musCgrad

def gradientsMuKernelParallel(mu,c,t,j,A,B,dict1):
    

    alphas = A[0]
    alpha0 = A[1]
    betas = B[0]
    beta0 = B[1]
    tj=t[j]
    inflectionKernel=dict1['inflection']
    
    if tj>0:
        iP=t[j-1]
        lj=max(0,j-support)
        temp=t[lj:j]
      
        inflectionPs=inflectionKernel+temp.reshape(-1,1)
        inflectionPs=inflectionPs.reshape(-1)
        inflectionPs=np.sort(inflectionPs)
        inflectionPs=inflectionPs[(inflectionPs>iP)*(inflectionPs<tj)]
        inflectionPs=np.append(iP,inflectionPs)
        inflectionPs=np.append(inflectionPs,tj)
        gradMu,gradKer,gradCn=gradientsMuKernelPart(mu,tj,iP,inflectionPs,temp,A,B)
        
        gradA=gradKer[0]
        gradA0=gradKer[1]
        gradB=gradKer[2]
        gradB0=gradKer[3]
        
        gradC=gradCn
        logPart=np.sum(nnMuKernel(tj,(tj-temp),mu,c,A,B))
        inverseLog=1/logPart
        Mn2=mu
            
        n1=np.maximum(betas*(tj-temp).reshape(1,-1)+beta0,0)
        dn1=(n1>0)
        n2=np.dot(alphas.T,n1)+alpha0
        
        dn2=((n2+Mn2).sum()>0)
        if dn2!=0:
            
            gradMu-=1*inverseLog*(dn2>0)*(j>0)

            gradA0-=(len(temp))*dn2*inverseLog
            gradA-=(np.sum(n1,axis=1)*dn2*inverseLog).reshape(-1,1)
            gradB-=(np.sum(alphas*(tj-temp).reshape(1,-1)*dn1,axis=1)*dn2*inverseLog).reshape(-1,1)
            gradB0-=(np.sum(alphas*dn1,axis=1)*dn2*inverseLog).reshape(-1,1)
            gradC-=(1*inverseLog)*(tj>0)
    else:
        
        gradA=gradB=gradB0=np.zeros((len(alphas),1))
        gradA0=gradMu=gradC=0
        
    return gradMu,gradA,gradA0,gradB,gradB0,gradC
        

            

def gradientsMuKernelPart(mu,tj,iP,inflectionPs,temp,A,B):
    alphas = A[0]
    alpha0 = A[1]
    betas = B[0]
    beta0 = B[1]

 
    lambdaValInfl=lambdaVs(temp,inflectionPs,mu,A,B)
    
    oIp=[]
    if len(lambdaValInfl<=0)!=0:
        oIp.append(inflectionPs[0])
        for j in range(1,len(inflectionPs)):
            if ((lambdaValInfl[j]<=0) and (lambdaValInfl[j-1]>0)) or ((lambdaValInfl[j]>0) and (lambdaValInfl[j-1]<=0)):
                if (oIp[-1]==inflectionPs[j-1]):
                    oIp.append(inflectionPs[j])
                else:
                    oIp.append(inflectionPs[j-1])
                    oIp.append(inflectionPs[j])
        oIp=np.array(oIp)
        outerInflections=outerIps(oIp,temp,mu,A,B)
        if len(outerInflections)!=0:
            inflectionPs=np.concatenate((inflectionPs,outerInflections))
            inflectionPs=np.sort(inflectionPs)

    gradMu=0
    gradA=np.zeros((len(alphas),1))*0
    gradB=np.zeros((len(alphas),1))*0
    gradB0=np.zeros((len(alphas),1))*0
    gradA0=0
    gradC=0
    for k in range(1,len(inflectionPs)):
        iP1=inflectionPs[k]
        iP2=inflectionPs[k-1]
        Mn2=mu
        
        n1=np.maximum(betas*(iP1-temp-epsilon).reshape(1,-1)+beta0,0)
        dn1=(n1>0)
  
        n2=(np.dot(alphas.T,n1)+alpha0).sum()
     
        dn2=((Mn2+n2)>0)
        
        if dn2!=0:
            gradMu+=(iP1-iP2)*(dn2>0)*(iP2>0)        
            temp1=(iP1-temp).reshape(1,-1)
            temp2=(iP2-temp).reshape(1,-1)
            ck1=(np.sum(temp1*(beta0+0.5*betas*temp1)*dn1,axis=1)).reshape(-1,1)
            ck2=(np.sum(temp2*(beta0+0.5*betas*temp2)*dn1,axis=1)).reshape(-1,1)
            gradA+= (ck1-ck2)*dn2
            gradA0+=(temp1-temp2).sum()*dn2
            gradB+=(np.sum((alphas*0.5*(temp1)**2-alphas*0.5*(temp2)**2)*(dn1),axis=1)*dn2).reshape(-1,1)
            gradB0+=(np.sum((alphas*temp1-alphas*temp2)*(dn1),axis=1)*dn2).reshape(-1,1)
            gradC+=(iP1-iP2)*(iP2>0)
    gradKer=[gradA,gradA0,gradB,gradB0]
    return gradMu,gradKer,gradC
    




# In[31]:


#Plotting for kernel
    
def plotKernels1(fac,A,B):
    temp1=np.arange(0,5,0.01)
    kernel=nnKernel(temp1,A,B)
    plt.plot(temp1/fac,kernel.reshape(-1)*fac)
    plt.xlim(0,5)
    plt.pause(0.0005)

def preCalculations(t,fac):
    global epsilon,support,nNeurons,dict1
    global mu,c,tmax
    epsilon = 1e-8
    support=20
  
    A=[]
    B=[]
    tmax=t[-1]
    nNeurons=32
    mu,c,A,B=initializeParams(nNeurons,fac,t,A,B)
    
    inflectionPoints(A,B)
    return mu,c,A,B

def sgdNeuralHawkes(t,fac,nEpochs=50,lr=0.01,kerneltype=0):        
    global t_val
    mu,c,A,B=preCalculations(t,fac)
    errorListVal=[]
    
    lr2 =lr
    lr_mu=lr
    beta_1 = 0.9
    beta_2 =0.999
  
    
    bestll = 1e8
    
    optimalParams=[A,B,mu,c,bestll]
  
    m_t_A = np.zeros([nNeurons,1])
    m_t_A0 =0
    m_t_B= np.zeros([nNeurons,1])
    m_t_B0= np.zeros([nNeurons,1])

    v_t_A = np.zeros([nNeurons,1])
    v_t_A0 =0
    v_t_B= np.zeros([nNeurons,1])
    v_t_B0= np.zeros([nNeurons,1])

    count = 0
    

    m_t_A0_Mu = 0
    v_t_A0_Mu = 0
    
    m_t_C=0
    v_t_C=0
    stopping_count=0

    splittrain=int(0.70*len(t))
    splittest =int(0.85*len(t))
    t_train=t[0:splittrain]
    t_val=t[splittrain:splittest]
    t_test=t[splittest:]
   
    bsize=max(int(len(t_train)/100),100)
    stop=0
    for epochs in range(1,nEpochs+1,1):
                   
        rsample = np.random.choice(len(t_train),len(t_train),replace = True)
        for i in range(1,len(rsample), bsize):
       
            count=count+1 
            
            A_grad,B_grad,mus1grad,musCgrad=gradientsMuKernel(mu,c,t_train,rsample[i:i+ bsize],A,B,dict1)
     
            
            
            m_t_A = beta_1*m_t_A + (1-beta_1)*A_grad[0]	#updates the moving averages of the gradient
            v_t_A = beta_2*v_t_A + (1-beta_2)*(A_grad[0]*A_grad[0])	#updates the moving averages of the squared gradient
            m_cap_A = m_t_A/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A = v_t_A/(1-(beta_2**count))		#calculates the bias-corrected estimates
            A[0] = A[0]-(lr*m_cap_A)/(np.sqrt(v_cap_A)+epsilon)
            
            
            m_t_A0 = beta_1*m_t_A0 + (1-beta_1)*A_grad[1]	#updates the moving averages of the gradient
            v_t_A0 = beta_2*v_t_A0 + (1-beta_2)*(A_grad[1]*A_grad[1])	#updates the moving averages of the squared gradient
            m_cap_A0 = m_t_A0/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A0 = v_t_A0/(1-(beta_2**count))		#calculates the bias-corrected estimates
            #A[1] = A[1]-(lr*m_cap_A0)/(np.sqrt(v_cap_A0)+epsilon)
            
                
            m_t_B = beta_1*m_t_B + (1-beta_1)*B_grad[0]	#updates the moving averages of the gradient
            v_t_B = beta_2*v_t_B + (1-beta_2)*(B_grad[0]*B_grad[0])	#updates the moving averages of the squared gradient
            m_cap_B = m_t_B/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_B= v_t_B/(1-(beta_2**count))		#calculates the bias-corrected estimates
            B[0] = B[0]-(lr2*m_cap_B)/(np.sqrt(v_cap_B)+epsilon)
            
            
            m_t_B0 = beta_1*m_t_B0 + (1-beta_1)*B_grad[1]	#updates the moving averages of the gradient
            v_t_B0 = beta_2*v_t_B0 + (1-beta_2)*(B_grad[1]*B_grad[1])	#updates the moving averages of the squared gradient
            m_cap_B0 = m_t_B0/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_B0 = v_t_B0/(1-(beta_2**count))		#calculates the bias-corrected estimates
            B[1] = B[1]-(lr2*m_cap_B0)/(np.sqrt(v_cap_B0)+epsilon)
            
            


            
            m_t_A0_Mu = beta_1*m_t_A0_Mu + (1-beta_1)*mus1grad[0]	#updates the moving averages of the gradient
            v_t_A0_Mu = beta_2*v_t_A0_Mu + (1-beta_2)*(mus1grad[0]*mus1grad[0])	#updates the moving averages of the squared gradient
            m_cap_A0_Mu = m_t_A0_Mu/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A0_Mu = v_t_A0_Mu/(1-(beta_2**count))		#calculates the bias-corrected estimates
            mu= mu-(lr_mu*m_cap_A0_Mu)/(np.sqrt(v_cap_A0_Mu)+epsilon)
            mu=max(mu,1e-3)
            
            
            m_t_C = beta_1*m_t_C + (1-beta_1)*musCgrad[0]	#updates the moving averages of the gradient
            v_t_C = beta_2*v_t_C + (1-beta_2)*(musCgrad[0]*musCgrad[0])	#updates the moving averages of the squared gradient
            m_cap_C = m_t_C/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_C = v_t_C/(1-(beta_2**count))		#calculates the bias-corrected estimates
            c= c-(lr_mu*0.01*m_cap_C)/(np.sqrt(v_cap_C)+epsilon)
            c=max(1e-5*fac,c*fac)
            c=min(1e-10*fac,c*fac)

            inflectionPoints(A,B)
            if count>=10:
            
                error=nnLoglikelihoodVal(np.arange(1,len(t_val),1),t_val,mu,c,A,B,dict1)
                if(bestll > error):
                    A1=A[0]
                    A2=A[1]
                    B1=B[0]
                    B2=B[1]
                    
                    optimalParams=[A1,A2,B1,B2,mu,c,error]
                    bestll=error
                    stopping_count=0  
        
                else:
                    stopping_count+=1
                print(i,epochs,bestll,error,mu*fac,c)   #iteration, -loglikelihood, bestloglik, currentpara, bestpara 
                count=0
       
                errorListVal.append(error)
        
 
        if stopping_count>=20:
            return optimalParams,errorListVal
        plotKernels1(fac,A,B)    

    return optimalParams,errorListVal
    

#Initialization

def initializeParams(nNeurons,fac,t,A,B):

    
    alphas = (np.random.uniform(0,1,nNeurons)).reshape(-1,1)*0.2
    alpha0 = np.random.uniform(0,1,1)*0
    betas=(np.random.uniform(0,1,nNeurons)).reshape(-1,1)*-0.3

    beta0 =(np.random.uniform(0,1,nNeurons)).reshape(-1,1)*0.3
    

    A.append(alphas)
    B.append(betas)
    A.append(alpha0)
    B.append(beta0)

    mu=len(t)/t[-1]
    c=np.random.uniform(0,1,1)[0]*0.01
    return mu,c,A,B






