import numpy as np
from operator import itemgetter
import time as tm
import RALF1FilterX as XFilter
import sys
import lfib1340 
from scipy.signal import savgol_filter

def RALF1FilterXV(dQ2):    
    Np=len(dQ2)
    Nf=len(dQ2[0])
       
    SdQ=np.mean(dQ2,0)  
    sSdQ=np.std(np.asarray(SdQ,float))
    for i in range(Np):
        SdQj_ = np.std(np.asarray(dQ2[i] - SdQ,float))
        SdQj__ = np.std(np.asarray(dQ2[i],float))            
        if SdQj__ >0. and sSdQ>0.:
            dQ2[i] = np.asarray(dQ2[i] +SdQ * ((SdQj_ - sSdQ)/ sSdQ ),np.float16)
        else:
            dQ2[i]=np.zeros(Nf,np.float16)        
    return dQ2

def RandomXV(Nfx):
    KK=3e6
    liiX=np.zeros(Nfx,float)
    for ii in range(3): 
        z=np.random.randint(Nfx)/KK           
        atim0=tm.time()        
        tm.sleep(z) 
        atim=tm.time()     
        dd=int((atim-atim0-z)*KK)
        zz=np.asarray(range(Nfx),float)/KK
        lfib1340.LFib1340(dd).shuffle(zz)   
        liiX=liiX+zz
            
    r2=np.zeros((2,Nfx),float)
    r2[0]= np.asarray(liiX[0:Nfx],float)
    r2[1]= np.asarray(range(Nfx),int)
    m=[[r2[j][l] for j in range(len(r2))] for l in range(len(r2[0]))]         
    m.sort(key=itemgetter(0))                  
    r2=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]  
    liiXX=np.asarray(r2[1],int)
    return liiXX

def filterFourierXV(arxx,arb,NNew,NChan):  
    Nfl=int(len(arb)/NChan)
    Nnl=NNew
    
    ar_=np.zeros(Nnl,float)
    farx=np.zeros(Nnl,float)
    
    az=int(np.floor(Nfl/Nnl))-1
    
    for l in range(NChan):        
        for i in range(az):
            for j in range(Nnl):
                ar_[j]=arb[Nfl-(az-i+1)*Nnl+j+Nfl*l]
            ar_=abs(np.fft.fft(ar_))
            for j in range(Nnl):
                farx[j]=max(farx[j],ar_[j])
    
    farx[0]=1e-32
    arxr=np.zeros(Nfl*NChan,float)   
    for l in range(NChan):       
        farxx=np.fft.fft(arxx[Nfl-Nnl+Nfl*l:Nfl+Nfl*l])    
        mfarxx=abs(farxx) 
        mfarxx[0]=1e-32
        srmfarxx=.62*np.mean(mfarxx[1:])
        farxxx=np.zeros(Nnl,complex)     
        for j in range(Nnl):
            if mfarxx[j]>srmfarxx:
                farxxx[j]=farxx[j]/mfarxx[j]*farx[j]            
            else:
                farxxx[j]=0        
        arxr[Nfl-Nnl+Nfl*l:Nfl+Nfl*l]=np.fft.ifft(farxxx).real  
        arxr[0+Nfl*l:Nfl-Nnl+Nfl*l]=arb[0+Nfl*l:Nfl-Nnl+Nfl*l].copy() 

    return arxr

def RALf1FiltrXV(args):
    NChan=int(args[1])
    NNew=int(args[2])
    Nhh=int(args[3])
    Nf=int(len(args)-4) 
       
    arr_bb=[]    
    for i in range(Nf):
        arr_bb.append(args[4+i])
    arr_bb=np.asarray(arr_bb,float)
            
    arr_b=arr_bb.copy() 
    Nf=int(arr_b.size/NChan)
    arr_bZ=[]
    for l in range(NChan):
        arr_bZ.append(arr_b[0+Nf*l:Nf-NNew+Nf*l])    
    arr_bZ=np.asarray(arr_bZ,float)
    D=np.std(arr_bZ)
    mn=np.mean(arr_bZ)
    arr_b=np.asarray(arr_bb,np.float16)    
        
    hh=0
    ann=0
     
    NNew=int(NNew*1.1)
    arr_bbx=[]
    while hh<Nhh:
        liiB=np.zeros(2*Nf*NChan,int)
        aa=RandomXV(Nf*NChan) 
        liiB[0:Nf*NChan]=aa
        liiB[Nf*NChan:2*Nf*NChan]=aa        
        
        liiC=np.zeros(2*(Nf+1)*NChan,int)
        aa=RandomXV((Nf+1)*NChan) 
        liiC[0:(Nf+1)*NChan]=aa
        liiC[(Nf+1)*NChan:2*(Nf+1)*NChan]=aa   
        
        liiD=RandomXV(Nf*NChan)
        liiE=RandomXV(Nf*NChan)
        
        r4=np.zeros(Nf*NChan,float)
        for l in range(NChan):            
            r4[Nf-NNew+Nf*l:Nf+Nf*l]=RandomXV(NNew)/NNew 
            r4[Nf-NNew+Nf*l:Nf+Nf*l]=D*(r4[Nf-NNew+Nf*l:Nf+Nf*l]/np.std(r4[Nf-NNew+Nf*l:Nf+Nf*l])/2+1e-6) 
                            
        r2=np.asarray(arr_b,np.float16)
        for l in range(NChan):                
            r2[Nf-NNew+Nf*l:Nf+Nf*l]=mn
        r2=r2-mn
        R4=np.asarray(r4,np.float16)
        K=NNew/(Nf+1)/NChan
        sz=int(NChan*Nf)
        liix=[[] for kk in range(Nf*NChan)]  

        line=1
        while line==1: 
            tm.sleep(0.2)
            anamef="fralf.tmp"
            try:
                fo = open(anamef, "r")
                line = int(fo.readline()) 
            except:
                fo = open(anamef, "w")
                fo.write(str(0)+'\n') 
                line=0                       
            fo.close()
            if line==0:
                fo = open(anamef, "w")
                fo.write(str(1)+'\n')
                fo.close()
                dQ3=np.zeros((sz,sz),np.float16)
                mDD=np.zeros((sz,sz),np.float16)
                fo = open(anamef, "w")
                fo.write(str(0)+'\n')
                fo.close() 
        
        for i in range(sz):    
            r1=liiB[int(liiD[i]):sz+int(liiD[i])]                                     
            liix[i].append(np.asarray(r1,int)) 
                                                 
            # r1=liiB[int(liiD[i]):Nf+int(liiD[i])].copy()
            # lfib1340.LFib1340(int(liiC[i])).shuffle(r1)                         
            # liix[i].append(r1)
                     
            dQ3[i]=r2[r1]
            for l in range(NChan):
                bb=np.asarray(liiC[np.asarray(np.arange(l+int(liiE[i]),l+sz+int(liiE[i]),sz/NNew),int)]*K,int)
                R4[Nf-NNew+Nf*l:Nf+Nf*l]=r4[Nf-NNew+Nf*l+bb]            
            # R4=r4.copy()
            # lfib1340.LFib1340(int(liiD[i])).shuffle(R4[Nf-NNew:])    
            mDD[i]=R4[r1]  
            tm.sleep(0.002)
             
        line=1
        while line==1: 
            tm.sleep(0.2)
            anamef="fralf.tmp"
            try:
                fo = open(anamef, "r")
                line = int(fo.readline()) 
            except:
                fo = open(anamef, "w")
                fo.write(str(0)+'\n') 
                line=0                       
            fo.close()
            if line==0:
                fo = open(anamef, "w")
                fo.write(str(1)+'\n')
                fo.close()
                # dQ3=np.reshape(np.asarray(dQ3,np.float16),(Nf*NChan,Nf*NChan))            
                # mDD=np.reshape(np.asarray(mDD,np.float16),(Nf*NChan,Nf*NChan))
                
                dQ3=(XFilter.RALF1FilterX(  dQ3-dQ3*(dQ3<0) +mDD,Nf,Nf,1,0)-
                     XFilter.RALF1FilterX(-(dQ3-dQ3*(dQ3>0))+mDD,Nf,Nf,1,0))
                fo = open(anamef, "w")
                fo.write(str(0)+'\n')
                fo.close()                
        
        del(mDD)       
        for i in range(sz):
            r1=np.asarray(liix[i],int)
            dQ3[i][r1]=dQ3[i][:]     
        aMx=np.max(dQ3,0)
        aMn=np.min(dQ3,0)        
        del(liix)
        del(dQ3)
        
        for l in range(NChan):                
            aMx[0+Nf*l:Nf+Nf*l]= savgol_filter(aMx[0+Nf*l:Nf+Nf*l], 11, 5)
            aMn[0+Nf*l:Nf+Nf*l]= savgol_filter(aMn[0+Nf*l:Nf+Nf*l], 11, 5)
        arr_bbbxxx=aMx + aMn  
        
        arr_bbbxxx=filterFourierXV(arr_bbbxxx,arr_b,NNew,NChan)
        
        ann=sum(np.isnan(arr_bbbxxx))
        if ann==0: 
            arr_bbx.append(arr_bbbxxx)           
            hh=hh+1
    
    arr_bbx=np.asarray(arr_bbx,np.float16).transpose()
    for l in range(NChan): 
        for ii in range(NNew):  
            arr_b[ii+Nf-NNew+Nf*l]=(max(arr_bbx[ii+Nf-NNew+Nf*l])+min(arr_bbx[ii+Nf-NNew+Nf*l]))/2        
    #arr_b=filterFourierV(arr_b,arr_b,NNew,NChan)
         
    return arr_b+mn

if __name__ == '__main__':
    RALf1FiltrXV(sys.argv)