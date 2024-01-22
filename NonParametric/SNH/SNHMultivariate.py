#!/usr/bin/env python
# coding: utf-8




import numpy as np
import matplotlib.pyplot as plt
support=20

optimalParams = []
epsilon = 1e-8
Dict_inf={}
Dict_con={}
Dict_integrate={}
Dict_gradient={}
dictdimP={}
mapping={}


def createMapAtoBIndex(a,b):
    mapAtoBIndex={}
    p1=0
    for x in range(len(a)):
        y=a[x]
        p=max(0,x)
        b1=b[p1:]
        if(max(b1[b1<y],default=-1)==-1):
            mapAtoBIndex[y] = None
        else:
            p1= p1+(np.where(b1==max(b1[b1<y])))[0][0]
            mapAtoBIndex[y] = p1    

    return mapAtoBIndex

#------------------------------ plots -------------------------------------------------#

def plotKernelsAll():

    dx=0.01
    tk = np.arange(0,5,dx)
    y1 = np.zeros([len(tk),totalD,totalD])
    y2 = np.zeros([len(tk),totalD,totalD])
    fig, ax = plt.subplots(totalD, totalD,sharex=True,sharey=True)
    
    for p in range(totalD):
        for k in range(totalD):
            alpha1=alpha[p,k]
            beta1=beta[p,k]
            y1[:,p,k] = alpha1*np.exp(-beta1*tk)    
            ax[p][k].plot(tk.reshape(-1), y1[:,p,k],'tab:blue')
            y2[:,p,k] = nnKernel(tk.reshape(1,-1),p,k)
            ax[p][k].plot(tk.reshape(-1),y2[:,p,k],'tab:orange')
            ax[p][k].set_title(r"$\phi_{%g,%g}(t)$" %(p,k))
            #ax[p][k].set_ylim(y2[0,p,k],y2[,p,k])
            ax[p][k].grid()
    
    plt.tight_layout()
    plt.pause(0.005)
    return


## ------------------------Code for Neural Hawkes starts here-------------------------##


def inflectionPoints():
    Dict_inf.clear()
    Dict_con.clear()
    for p in range(totalD):
        for k in range(totalD):
            alphas = Alphas[:,p,k]
    
            betas = Betas[:,p,k]
            beta0 = Beta0[:,p,k]
    
            div = betas+epsilon*(np.abs(betas)<epsilon) # potentially error prone
            x = -beta0/div
            interestX1 = x*(x>0)
            alwaysInclude = (x<=0)*(betas>0) #dont change
            alwaysExclude = (x<=0)*(betas<0)
            tempX = x*(~alwaysInclude)*(~alwaysExclude)
            interestX1 = tempX[interestX1>0]
            interestX = np.sort(interestX1)
            interestX = np.append(0,interestX)
        
            Dict_inf[p,k]= interestX
   
            con = alphas*betas
            Dict_con[p,k] = con
    return


def nnIntegratedKernel(x,p,k):
    x = x.reshape(-1)
    alphas = Alphas[:,p,k].reshape(-1,1)
    alpha0 = Alpha0[:,p,k].reshape(-1,1)
    betas = Betas[:,p,k].reshape(-1,1)
    beta0 = Beta0[:,p,k].reshape(-1,1)
    const1 = Dict_con[p,k]
    interestX = Dict_inf[p,k]
    precalculate_integrate(p,k)
    
    y = np.zeros([1,max(x.shape)])
    
    for i in range(0,max(x.shape)):
        xi = x[i]
        if(xi>0):
            iP = max(interestX[interestX<xi])
            n1 = betas*(xi-epsilon) + beta0
            dn1 = (n1>0)
            const =np.dot(const1.T,dn1)
            
            
            term1 = nnKernel(xi,p,k)*((const!=0)+xi*(const==0))
            term2 = nnKernel(iP,p,k)*((const!=0)+iP*(const==0))
            
            const = (const)*(const!=0)+(const==0)*1.0
            
            prev_term = Dict_integrate[p,k][iP]
            
            y[0,i] = prev_term + ((term1-term2)/(const))
        
       
               

    return y

def nnKernel(x,p,k):
    alphas = Alphas[:,p,k].reshape(-1,1)
    alpha0 = Alpha0[:,p,k]
    betas = Betas[:,p,k].reshape(-1,1)
    beta0 = Beta0[:,p,k].reshape(-1,1)
    n1 = np.maximum(np.dot(betas,x) + beta0,0.)
    y = np.dot(alphas.T,n1) + alpha0
    
    y = np.exp(y)
    return y   



def initializeParams(nNeurons,t,fac):
    
    
    for p in range(totalD):
        for k in range(totalD):
            Alphas[:,p,k] = (np.random.uniform(-1,1,nNeurons)).reshape(-1)*2/fac
            Alpha0[:,p,k] = -np.random.uniform(0,1,1)*0.2
            Betas[:,p,k] = (np.random.uniform(0,1,nNeurons)).reshape(-1)*-0.1/fac
            Beta0[:,p,k] = np.random.uniform(0,1,nNeurons).reshape(-1)*0.1
            
        mu1[p] = len(t[p])/t[p][-1]*fac
    return mu1


def precalculate_integrate(p,k):
    alphas = Alphas[:,p,k].reshape(-1,1)
    alpha0 = Alpha0[:,p,k].reshape(-1,1)
    betas = Betas[:,p,k].reshape(-1,1)
    beta0 = Beta0[:,p,k].reshape(-1,1)
    iP = Dict_inf[p,k]
    const1 = Dict_con[p,k]
    Dict_integrate[p,k].clear()
    Dict_integrate[p,k][0]=0
    y = 0
    for index in range(1,len(iP)):
        n1 = betas*(iP[index]-epsilon) + beta0
        dn1 = (n1>0)
        const =np.dot(const1.T,dn1)
            
            
        term1 = nnKernel(iP[index],p,k)*((const!=0)+iP[index]*(const==0))
        term2 = nnKernel(iP[index-1],p,k)*((const!=0)+iP[index-1]*(const==0))
            
        const = (const)*(const!=0)+(const==0)*1.0
              
        y= y + ((term1-term2)/(const))
        Dict_integrate[p,k][iP[index]]=y
    return



def precalculate_gradient(p,k):
    alphas = Alphas[:,p,k].reshape(-1,1)
    alpha0 = Alpha0[:,p,k].reshape(-1,1)
    betas = Betas[:,p,k].reshape(-1,1)
    beta0 = Beta0[:,p,k].reshape(-1,1)
    gradA = alphas*0
    gradB1 = gradA*0
    gradB0 = gradB1*0
    gradA0=0
    iP = Dict_inf[p,k]
    
    const1 = Dict_con[p,k]
    Dict_gradient[p,k].clear()
    Dict_gradient[p,k][0]=list([gradA0,gradA,gradB1,gradB0])
    for index in range(1,len(iP)):
            
            n0pr = betas*(iP[index-1]+epsilon)+beta0
            n1pr = betas*(iP[index]-epsilon) + beta0
            dn1 = (n1pr>0)
            dn0 = (n0pr>0)
            const =np.dot(const1.T,dn1)
            indicator = const==0
            const = const*(const!=0)+1*(const==0)
            n0 = betas*(iP[index-1])+beta0
            n1 = betas*(iP[index]) + beta0
            
            fac1 = nnKernel(iP[index],p,k)
            fac2 = nnKernel(iP[index-1],p,k)
            gradA0 = gradA0 + ((1/const)*(fac1-fac2))*(~indicator)+(fac1)*(indicator)*(iP[index]-iP[index-1])
            gradA = gradA -((1/(const*const))*(betas*dn1)*(fac1-fac2))*(~indicator)
            gradA = gradA + ((1/const)*(fac1*np.maximum(n1,0)-fac2*np.maximum(n0,0)))*(~indicator)
            gradB1 = gradB1 -((1/(const*const))*(alphas*dn1)*(fac1-fac2))*(~indicator)
            gradB1 = gradB1 + ((1/const)*(alphas)*(fac1*iP[index]*dn1-fac2*iP[index-1]*dn0))*(~indicator)
            gradB0 = gradB0+ ((1/const)*((alphas)*(fac1*dn1-fac2*dn0)))*(~indicator)
            Dict_gradient[p,k][iP[index]]= list([gradA0,gradA,gradB1,gradB0])
            
    return


def gradientNNIntegratedKernel(tend,p,k):
    
    alphas = Alphas[:,p,k].reshape(-1,1)
    alpha0 = Alpha0[:,p,k].reshape(-1,1)
    betas = Betas[:,p,k].reshape(-1,1)
    beta0 = Beta0[:,p,k].reshape(-1,1)
    
    gradA = alphas*0
    gradB1 = gradA*0
    gradB0 = gradB1*0
    gradA0=0
    
    if(tend>0):
        const1 = Dict_con[p,k]
        interestX = Dict_inf[p,k]
            
        iP = max(interestX[interestX<tend])
        n1pr = betas*(tend-epsilon) + beta0
        n0pr = betas*(tend+epsilon)+beta0
           
        dn1 = (n1pr>0)
        dn0 = (n0pr>0)
        const =np.dot(const1.T,dn1)
        indicator = const==0
        const = const*(const!=0)+1*(const==0)
        n0 = betas*(iP)+beta0
        n1 = betas*(tend) + beta0
        fac1 = nnKernel(tend,p,k)
        fac2 = nnKernel(iP,p,k)
        gradients = Dict_gradient[p,k][iP]
            
            
        gradA0 = gradA0+gradients[0] + ((1/const)*(fac1-fac2))*(~indicator)+(fac1)*(indicator)*(tend-iP)
            
        gradA = gradA + gradients[1] -((1/(const*const))*(betas*dn1)*(fac1-fac2))*(~indicator)
        gradA = gradA + ((1/const)*(fac1*np.maximum(n1,0)-fac2*np.maximum(n0,0)))*(~indicator)
            
        gradB1 = gradB1+gradients[2] -((1/(const*const))*(alphas*dn1)*(fac1-fac2))*(~indicator)
        gradB1 = gradB1+ ((1/const)*(alphas)*(fac1*tend*dn1-fac2*iP*dn0))*(~indicator)
    
        gradB0 =gradB0+ gradients[3] + ((1/const)*((alphas)*(fac1*dn1-fac2*dn0)))*(~indicator)
    return list([gradA0,gradA,gradB1,gradB0])



    
def gradientNNKernel(temp,p,k):
    alphas = Alphas[:,p,k].reshape(-1,1)
    alpha0 = Alpha0[:,p,k]
    betas = Betas[:,p,k].reshape(-1,1)
    beta0 = Beta0[:,p,k].reshape(-1,1)
    
    gradA = alphas*0
    gradB1 = gradA*0
    gradB0 = gradB1*0
    gradA0=0
    fac1 = nnKernel(temp.reshape(1,-1),p,k)
    
    n1 = np.dot(betas,temp.reshape(1,-1)) + beta0
    gradA = gradA + np.sum(fac1*np.maximum(n1,0),axis=1).reshape(-1,1)
        
    gradA0 = gradA0 + np.sum(fac1)
        
    gradB1 = gradB1 + np.sum(fac1*(n1>0)*alphas*temp.reshape(1,-1),axis=1).reshape(-1,1)
    gradB0 = gradB0 + np.sum(fac1*(n1>0)*alphas,axis=1).reshape(-1,1)
    return list([gradA0,gradA,gradB1,gradB0])    




     
def gradientNetwork(iArray,t):
    nNeurons=len(Alphas[:,0,0])
    gradA0 = np.zeros([1,totalD,totalD])
    gradA = np.zeros([nNeurons,totalD,totalD])
    gradB1 = np.zeros([nNeurons,totalD,totalD])
    gradB0 = np.zeros([nNeurons,totalD,totalD])
    grad_mu = np.zeros(totalD)
    nSamples = max(iArray.shape)
    
    for p in range(totalD):
        for k in range(totalD):
            precalculate_gradient(p,k)
    
    for i in range(0,nSamples,1):
        p = int(iArray[0,i])
        index = int(iArray[1,i])
        
        if index>0:
            
            for k in range(totalD):
                tend=t[p][-1]-t[p][index]

                if (tend>=0):
                    out = gradientNNIntegratedKernel(tend,k,p)
                    gradA0[:,k,p]=gradA0[:,k,p]+out[0]
                    gradA[:,k,p]=gradA[:,k,p]+out[1].reshape(-1)
                    gradB1[:,k,p]=gradB1[:,k,p]+out[2].reshape(-1)
                    gradB0[:,k,p]=gradB0[:,k,p]+out[3].reshape(-1)
                    
            decayFactor=0
            temp=[[] for j in range(totalD)]
            for k in range(totalD):
                if p==k:
                    li = max(index-20,0)
                    temp1 = t[p][index]-t[p][li:index]
                    temp[p] = temp1
                else:
                    jT = mapping[p,k].get(t[p][index])
                    if(jT != None):
                        j = (jT)
                        lj = max(j-20,0)
                        temp1 = t[p][index]-t[k][lj:j+1]
                        temp[k]=temp1
                decayFactor += np.sum(nnKernel(temp1.reshape(1,-1),p,k))
                lam = mu1[p]+decayFactor
                invLam = (1/lam)
    
            for k in range(totalD):
                out =  gradientNNKernel(temp[k],p,k)
        
                gradA0[:,p,k]=gradA0[:,p,k] -invLam*out[0]
                gradA[:,p,k]=gradA[:,p,k] -invLam*out[1].reshape(-1)
                gradB1[:,p,k]=gradB1[:,p,k]-invLam*out[2].reshape(-1)
                gradB0[:,p,k]=gradB0[:,p,k] -invLam*out[3].reshape(-1)
        

            grad_mu[p]=grad_mu[p]+((t[p][index]-t[p][index-1])-(1/lam))*(t[p][index-1]>0)*((index-1)>0) 
        
    gradA0 = gradA0/(nSamples)
    gradA = gradA/nSamples
    gradB1 = gradB1/nSamples
    gradB0 = gradB0/nSamples
    grad_mu =grad_mu/nSamples
        
    return list([gradA0,gradA,gradB1,gradB0,grad_mu])
     

     


def nnLoglikelihood(t):
    ll=0
    integrated=0
    for p in range(0,totalD,1):
        tend = (t[p][-1]-t[p][:])
        a = np.sum(nnIntegratedKernel(tend.reshape(1,-1),p,p))
        
        for k in dictdimP[p]:
            tend=(t[p][-1]-t[k][:])
            tend=tend*(tend>=0)
            a+=np.sum(nnIntegratedKernel(tend.reshape(1,-1),p,k))
        
        ll = ll+ mu1[p]*(t[p][-1]-t[p][0])+a
        integrated+=ll
       
        ll = ll-np.log(mu1[p])
        
        tp=t[p]
        for i in range(1,len(tp),1):
            li = max(i-50,0)
            temp1 = tp[i]-tp[li:i]
            decayFactor = np.sum(nnKernel(temp1.reshape(1,-1),p,p))
            for k in dictdimP[p]:
                jT = mapping[p,k].get(tp[i])
                if(jT != None):
                    j =(jT).item()
                    lj = max(j-50,0)
                    temp1 = tp[i]-t[k][lj:j+1]
                decayFactor+= np.sum(nnKernel(temp1.reshape(1,-1),p,k))
            logLam = -np.log(mu1[p]+decayFactor)
            
            ll = ll+logLam
    print("integrated Kernel",integrated)
       
    return ll


def sgdNeuralHawkesBiVariate(t,fac,nNeurons=50,nEpochs=50,lr=0.005):
    
    global optimalParams
    global Alpha0
    global Alphas
    global Betas
    global Beta0
    global mu1
    global totalD
    global tmax
    totalD=len(t)
    Alphas = np.zeros([nNeurons,totalD,totalD])
    Alpha0 = -np.zeros([1,totalD,totalD])
    Betas = np.zeros([nNeurons,totalD,totalD])
    Beta0 = np.zeros([nNeurons,totalD,totalD])
    mu1 = np.zeros(totalD)
    initializeParams(nNeurons,t,fac)
    for j in range(totalD):
        if t[j][0]!=0:
            t[j]=np.insert(t[j],0,0)
        if t[j][0]==t[j][1]==0:
            t[j]=np.delete(t[j],0)

    for p in range(totalD):
        for k in range(totalD):
            Dict_integrate[p,k] = {}  #nested dictionary for all networks
            Dict_gradient[p,k] = {}

        
    dimensions=np.arange(0,totalD,1)
    for i in range(totalD):
        dictdimP[i]=np.delete(dimensions,i)
    for i in range(totalD):
        mapping[i,i]=createMapAtoBIndex(t[i],t[i])
        for j in (dictdimP[i]):
            mapping[i,j]=createMapAtoBIndex(t[i],t[j])
    
    tmax=0
    for p in range(totalD):
        tmax=max(tmax,t[p][-1])
    totalLength=0

    for j in range(len(t)):
        totalLength+=len(t[j])
    combinedT=np.ones((totalLength,2))

    start=0
    for j in range(len(t)):
        end=start+len(t[j])
        combinedT[start:end,0]=t[j]
        combinedT[start:end,1]=np.ones(end-start)*j
        start=end
        
    sortedT=combinedT[combinedT[:, 0].argsort()]
    train_len=int(totalLength*0.7)
    test_len=int(totalLength*0.85)
    train1=sortedT[:train_len]
    val1=sortedT[train_len:test_len]
    test1=sortedT[test_len:]
    trainT=[]
    valT=[]
    testT=[]

    
    for j in range(len(t)):
        trainT.append(train1[train1[:,1]==j][:,0])
      
        valT.append(val1[val1[:,1]==j][:,0])
        
        testT.append(test1[test1[:,1]==j][:,0])

    
    lr2 =lr*0.1
    lr_mu = lr*0.1
    
    beta_1 = 0.9
    beta_2 =0.999
  
    
    bestll = 1e8
    neg_ll = []
    
    optimalParams = list([Alpha0,Alphas,Betas,Beta0])
    
    m_t_A = np.zeros([nNeurons,totalD,totalD])
    m_t_A0 =np.zeros([1,totalD,totalD])
    m_t_B= np.zeros([nNeurons,totalD,totalD])
    m_t_B0= np.zeros([nNeurons,totalD,totalD])
    m_t = np.zeros(totalD)
    v_t_A = np.zeros([nNeurons,totalD,totalD])
    v_t_A0 =np.zeros([1,totalD,totalD])
    v_t_B= np.zeros([nNeurons,totalD,totalD])
    v_t_B0= np.zeros([nNeurons,totalD,totalD])
    v_t = np.zeros(totalD)
    count = 0
    print(totalLength,"number of timepoints")
    train_len=0
    for j in range(len(trainT)):
        train_len+=len(trainT[j])
    

    tCompressed=np.zeros((2,train_len))
    length=0
    for j in range(len(trainT)):
        length1=length
        length+=len(trainT[j])
        tCompressed[0,length1:length]=j
        tCompressed[1,length1:length]=np.arange(0,length-length1,1)
    
    for epochs in range(1,nEpochs+1,1):
        inflectionPoints()
        rsample = np.random.choice(length,length,replace = False)
        for i in range(0,len(rsample),50):
            count=count+1 
            grad = gradientNetwork(tCompressed[:,rsample[i:i+50]],trainT)
            
            m_t = beta_1*m_t + (1-beta_1)*grad[4]	#updates the moving averages of the gradient
            v_t = beta_2*v_t + (1-beta_2)*(grad[4]*grad[4])	#updates the moving averages of the squared gradient
            m_cap = m_t/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap = v_t/(1-(beta_2**count))		#calculates the bias-corrected estimates
            mu1 = mu1-(lr_mu*m_cap)/(np.sqrt(v_cap)+epsilon)
            mu1 = np.maximum(mu1,1e-5)
           
           
            m_t_A0 = beta_1*m_t_A0 + (1-beta_1)*grad[0]	#updates the moving averages of the gradient
            v_t_A0 = beta_2*v_t_A0 + (1-beta_2)*(grad[0]*grad[0])	#updates the moving averages of the squared gradient
            m_cap_A0 = m_t_A0/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A0 = v_t_A0/(1-(beta_2**count))		#calculates the bias-corrected estimates
            Alpha0 = Alpha0-(lr*m_cap_A0)/(np.sqrt(v_cap_A0)+epsilon) 
                         
                   
            m_t_A = beta_1*m_t_A + (1-beta_1)*grad[1]	#updates the moving averages of the gradient
            v_t_A = beta_2*v_t_A + (1-beta_2)*(grad[1]*grad[1])	#updates the moving averages of the squared gradient
            m_cap_A = m_t_A/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A = v_t_A/(1-(beta_2**count))		#calculates the bias-corrected estimates
            Alphas = Alphas-(lr*m_cap_A)/(np.sqrt(v_cap_A)+epsilon)
            
            

            
                
            m_t_B = beta_1*m_t_B + (1-beta_1)*grad[2]	#updates the moving averages of the gradient
            v_t_B = beta_2*v_t_B + (1-beta_2)*(grad[2]*grad[2])	#updates the moving averages of the squared gradient
            m_cap_B = m_t_B/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_B= v_t_B/(1-(beta_2**count))		#calculates the bias-corrected estimates
            Betas = Betas-(lr2*m_cap_B)/(np.sqrt(v_cap_B)+epsilon)
            
            
            m_t_B0 = beta_1*m_t_B0 + (1-beta_1)*grad[3]	#updates the moving averages of the gradient
            v_t_B0 = beta_2*v_t_B0 + (1-beta_2)*(grad[3]*grad[3])	#updates the moving averages of the squared gradient
            m_cap_B0 = m_t_B0/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_B0 = v_t_B0/(1-(beta_2**count))		#calculates the bias-corrected estimates
            Beta0 = Beta0 -(lr2*m_cap_B0)/(np.sqrt(v_cap_B0)+epsilon)
          
            inflectionPoints()
            
        error=nnLoglikelihood(valT)
        neg_ll.append(error)
            #bestpara = para*(bestll>=error)+bestpara*(bestll<error)
            
        if(bestll > error):
            optimalParams = list([Alpha0,Alphas,Betas,Beta0,mu1])
                
               
            
        bestll = min(bestll,error)
        print(i,epochs,bestll,error,mu1)   #iteration, -loglikelihood, bestloglik, currentpara, bestpara
        
        #plotKernelsAll()
    return optimalParams,neg_ll,bestll
            
            
def optimalKernel(p,k,t):
    alphas = optimalParams[1][:,p,k]
    alpha0 = optimalParams[0][:,p,k].reshape(-1,1)
    betas = optimalParams[2][:,p,k].reshape(-1,1)
    beta0 = optimalParams[3][:,p,k].reshape(-1,1)
    
    n1 = np.maximum(np.dot(betas,t.reshape(1,-1)) + beta0,0.)
    y = np.dot(alphas.T,n1) + alpha0
    
    y = np.exp(y)
    return y    





