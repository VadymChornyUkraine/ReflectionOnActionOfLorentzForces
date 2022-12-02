import numpy as np
from operator import itemgetter
import time as tm
import RALF1FilterX as XFilter
import scipy.interpolate as scpyi 
import mersenne_twister as gen
import sys
import lfib1340 
from scipy.signal import savgol_filter

def RandomXY(Nfx):
    KK=3e6
    liiX=np.zeros(Nfx,float)
    for ii in range(3): 
        z=np.random.randint(Nfx)/KK           
        atim0=tm.time()        
        tm.sleep(z) 
        atim=tm.time()         
        zz=np.random.uniform(low=Nfx, high=Nfx*2, size=(Nfx,))/KK
        for i in range(Nfx):
            atim0=tm.time() 
            tm.sleep(zz[i]) 
            l=int(i*1)
            atim=tm.time()            
            liiX[l]=liiX[l]+atim-atim0-zz[l]
            
    r2=np.zeros((2,Nfx),float)
    r2[0]= (liiX[0:Nfx]).copy()
    r2[1]= np.asarray(range(Nfx),int).copy()
    m=[[r2[j][l] for j in range(len(r2))] for l in range(len(r2[0]))]         
    m.sort(key=itemgetter(0))                  
    r2=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]  
    liiX=np.asarray(r2[1].copy(),float)
    return liiX

def randomX(Nfx):
    KK=1e6
    liiX=np.zeros(Nfx,float)
    for ii in range(3):            
        atim0=tm.time() 
        z=np.random.randint(Nfx)/KK
        tm.sleep(z) 
        atim=tm.time()
        delt=(atim-atim0-z)*KK            
        genera = gen.mersenne_rng(int(delt))
        z=genera.get_random_number()/KK/KK
        i=0
        while i<Nfx:
            atim0=tm.time() 
            z=genera.get_random_number()/KK/KK#np.random.randint(Nf)/KK
            tm.sleep(z) 
            atim=tm.time()
            i=i+1
            liiX[i-1]=liiX[i-1]+atim-atim0-z
            
    r2=np.zeros((2,Nfx),float)
    r2[0]= (liiX[0:Nfx]).copy()
    r2[1]= np.asarray(range(Nfx),int).copy()
    m=[[r2[j][l] for j in range(len(r2))] for l in range(len(r2[0]))]         
    m.sort(key=itemgetter(0))                  
    r2=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]  
    liiX=np.asarray(r2[1].copy(),float)
    return liiX

def filterFourier(arxx,arb):  
    Nnl=len(arxx)
    Nfl=len(arb)
    ar_=np.zeros(Nnl,float)
    farx=np.zeros(Nnl,float)
    az=int(np.floor(Nfl/Nnl))-1
        
    for i in range(az):
        for j in range(Nnl):
            ar_[j]=arb[Nfl-(az-i+1)*Nnl+j]
        ar_=abs(np.fft.fft(ar_))
        for j in range(Nnl):
            farx[j]=max(farx[j],ar_[j])
    
    farx[0]=1e-32
    srfarx=.62*np.mean(farx[1:])
    
    farxx=np.fft.fft(arxx)    
    mfarxx=abs(farxx) 
    mfarxx[0]=1e-32
    srmfarxx=.62*np.mean(mfarxx[1:])
    farxxx=np.zeros(Nnl,complex)
    arxr=np.zeros(Nfl,float)
    
    for j in range(Nnl):
        if mfarxx[j]>srmfarxx and farx[j]>srfarx:
            farxxx[j]=farxx[j]/mfarxx[j]*farx[j]            
        else:
            farxxx[j]=0
    
    arxr[Nfl-Nnl:Nfl]=np.fft.ifft(farxxx).real  
    arxr[0:Nfl-Nnl]=arb[0:Nfl-Nnl].copy() 
    
    #arxr[Nfl-Nnl:Nfl]=arxr[Nfl-Nnl:Nfl]-arxr[Nfl-Nnl]+arxr[Nfl-Nnl-1]        
    return arxr

def RALf1FiltrX(args):
    if len(args)>1:
        NChan=int(args[1])
        NNew=int(args[2])
        Nhh=int(args[3])
        Nf=int(len(args)-4)        
        arr_bb=[]    
        for i in range(Nf):
            arr_bb.append(args[4+i])
        arr_bb=np.asarray(arr_bb,float)

    else:
        Nhh=2
        NChan=1
        file=open("anum.tmp",'r')
        NNew=int(file.readline())
        ss=file.readline()
        file.close()
        aname = "datapy%s.tmp"%ss
        file=open(aname,'r')
        arr_bb=np.asarray(file.readlines(),float).copy()
        file.close()
        
    ann=1

    arr_b=arr_bb.copy() 
    Nf=arr_b.size

    D=np.std(arr_b)
    
    tSp=1
    hh=0
    ann=0
     
    NNew=int(NNew*1.1)
    arr_bbx=[]
    while hh<Nhh:
        liiB=np.zeros(2*Nf,float)
        aa=RandomXY(Nf) 
        liiB[0:Nf]=aa
        liiB[Nf:2*Nf]=aa        
        
        liiC=RandomXY(Nf)
        liiD=RandomXY(Nf)
        
        r4=np.zeros(Nf,float)
        r4[Nf-NNew:]=RandomXY(NNew)/NNew 
        r4=D*(r4/np.std(r4[Nf-NNew:])/2+1e-6) 

        mn=np.mean(arr_b[0:Nf-NNew])
        liix=np.zeros((Nf,Nf*tSp),float)
        dQ3=np.zeros((Nf,Nf*tSp),float)   
        mDD=np.zeros((Nf,Nf*tSp),float)   
             
        for i in range(Nf):                                                     
            r1=liiB[int(liiD[i]):Nf+int(liiD[i])].copy()
            lfib1340.LFib1340(int(liiC[i])).shuffle(r1)
            ge=scpyi.interp1d(np.asarray(range(Nf),float),r1)                              
            liix[i]=ge(np.linspace(0,Nf-1,Nf*tSp)) 
            
            r2=arr_b.copy()     
            r2[Nf-NNew:]=mn               
            ge=scpyi.interp1d(np.asarray(range(Nf),float) ,r2,kind='cubic')
            dQ3[i]=ge(liix[i])    
            
            R4=r4.copy()
            lfib1340.LFib1340(int(liiD[i])).shuffle(R4[Nf-NNew:])    
            ge=scpyi.interp1d(np.asarray(range(Nf),float) ,R4,kind='cubic') 
            mDD[i]=ge(liix[i])             
        
        dQ3A=dQ3-mn        
        dQ3B=dQ3A-dQ3A*np.asarray(dQ3A<0,int)   
        dQ2X=XFilter.RALF1FilterX(dQ3B+mDD,Nf,Nf*tSp,1,0)
        dQ3C=-(dQ3A-dQ3A*np.asarray(dQ3A>0,int))   
        dQ2Y=-XFilter.RALF1FilterX(dQ3C+mDD,Nf,Nf*tSp,1,0)
        dQ2X=(dQ2X+dQ2Y)
        dQ2Y=dQ2X.copy()
    
        dQ5mx=np.zeros((Nf,Nf),float)
        dQ5mn=np.zeros((Nf,Nf),float)
        r4=np.zeros((3,Nf*tSp),float)
        for i in range(Nf):
            r4[0]= np.asarray(liix[i],int).copy()
            r4[1]= (dQ2X[i]).copy()
            r4[2]= (dQ2Y[i]).copy()
            m=[[r4[j][l] for j in range(len(r4))] for l in range(len(r4[0]))]         
            m.sort(key=itemgetter(0))                  
            r4=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]         

            anum1=-1;
            for j in range(Nf): 
                jjj0=anum1;
                anum0=max(0,min(jjj0,Nf*tSp-1));
                while jjj0<Nf*tSp-1 and int(r4[0][anum0])<j:
                    jjj0=jjj0+1;
                    anum0=max(0,min(jjj0,Nf*tSp-1));
                jjj1=anum0+1;
                anum1=max(0,min(jjj1,Nf*tSp));
                while jjj1<Nf*tSp and int(r4[0][anum1])<j+1:
                    jjj1=jjj1+1;
                    anum1=max(0,min(jjj1,Nf*tSp));
                dQ5mx[i][j]=max(r4[1][anum0:anum1])
                dQ5mn[i][j]=min(r4[2][anum0:anum1])
                
        dQ5mx=dQ5mx.transpose()
        dQ5mn=dQ5mn.transpose()
        
        dQ5mx=dQ5mx-dQ5mx*(dQ5mx<0)
        dQ5mn=dQ5mn-dQ5mn*(dQ5mn>0)
    
        aMx=np.zeros(Nf,float)
        aMn=np.zeros(Nf,float)
        
        for i in range(Nf):
            aMx[i]=max(dQ5mx[i])  
            aMn[i]=min(dQ5mn[i])          
            
        aMx= savgol_filter(aMx, 35, 5)
        aMn= savgol_filter(aMn, 35, 5)
        arr_bbbxxx=aMx + aMn  
        arr_bbbxxx=filterFourier(arr_bbbxxx[Nf-NNew:],arr_b)
        
        ann=sum(np.isnan(arr_bbbxxx))
        if ann==0: 
            arr_bbx.append(arr_bbbxxx)           
            hh=hh+1
    
    arr_bbx=np.asarray(arr_bbx,float).transpose()
    for ii in range(NNew):  
        arr_b[ii+Nf-NNew]=(max(arr_bbx[ii+Nf-NNew])+min(arr_bbx[ii+Nf-NNew]))/2
    
    arr_b=filterFourier(arr_b[Nf-NNew:],arr_b)+mn
         
    if len(args)==1:   
        fileout=open("rezpy%s.tmp"%ss,'w')
        for i in range(Nf):
            fileout.write(str(arr_b[i])+'\n')  
        fileout.close() 
    return arr_b

if __name__ == '__main__':
    RALf1FiltrX(sys.argv)