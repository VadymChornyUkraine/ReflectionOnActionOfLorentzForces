import numpy as np
def RALF1FilterX(dQ2,Np,Nf,key,key2):
    if key>0:     
        SdQj=np.ones(Np,float)   
        znakSdQj=np.ones(Np,float) 
        if key2>0:
            for i in range(Np):
                znakSdQj[i]=np.mean(dQ2[i])        
            znakSdQj=((znakSdQj<0)-0.5)*2*(-1)
        for i in range(Np): 
            #SdQj[i]=np.std(dQ2[i])
            dQ2[i] = dQ2[i] * znakSdQj[i]            
            if SdQj[i]==0:
                dQ2[i]=np.zeros(Nf,float)
            else:
                dQ2[i]=dQ2[i] / SdQj[i]
   
        SdQ=np.zeros(Nf,float)         
        for j in range(Nf):
            SdQ[j]=np.mean(dQ2[:,j])  
        sSdQ=np.std(SdQ)
        for i in range(Np):
            SdQj_ = np.std(dQ2[i] - SdQ)
            SdQj__ = np.std(dQ2[i])            
            if SdQj__ >0 and sSdQ>0:
                dQ2[i] = dQ2[i] +SdQ * (SdQj_ / sSdQ - 1)
                dQ2[i] = dQ2[i] *znakSdQj[i] * SdQj[i]#* (1+sSdQ / SdQj__) / 2 
            else:
                dQ2[i]=np.zeros(Nf,float)        
    return dQ2