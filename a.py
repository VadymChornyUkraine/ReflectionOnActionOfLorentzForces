import hickle as hkl
import numpy as np
from RALf1FiltrVID import filterFourierQ

lSrez=0.99
arrrxx=hkl.load("ralfrez.rlf2")
# i=0 
# arrrxx_=[]
# for ii in arrrxx:
#     if not i==4:
#         arrrxx_.append(ii)
#     i=i+1
#hkl.dump(arrrxx_,"ralfrez.rlf2")
try:
    ZDat=np.asarray(arrrxx,float)#.transpose()
    anI=len(ZDat)
    for ii in range(3):       
        ar0x=np.median(ZDat,axis=0)
        ar0x_=.4*(np.median(abs((ZDat)-(ar0x)),axis=0))
            
        lnn=len(ZDat[0])
        NNew=int(.35*lnn)
        for i in range(anI):    
            for j in range(lnn):    
                if not abs(ZDat[i,j]-(ar0x[j]))<=ar0x_[j]:     
                    if ZDat[i,j]<((ar0x[j])-ar0x_[j]):            
                        ZDat[i,j]=(ar0x[j])-ar0x_[j]
                    else:
                        if ZDat[i,j]>((ar0x[j])+ar0x_[j]):
                            ZDat[i,j]=(ar0x[j])+ar0x_[j]
        anI=anI
        P=np.zeros(3,float)
        for i in range(anI):
            dd=ZDat[i][lnn-NNew:].copy()                         
            x=ar0x[lnn-NNew:lnn-NNew+int(lSrez*(NNew-(lnn-len(ar0x))))].copy()
            ZDat[i][lnn-NNew:]=filterFourierQ(ZDat[i],(ar0x),NNew,1)[lnn-NNew:]
            P[0:2]=np.polyfit(x,ZDat[i][lnn-NNew:lnn-NNew+int(lSrez*(NNew-(lnn-len(ar0x))))],1)
            if not P[0]>0:
                P[0:2]=np.polyfit(dd,ZDat[i][lnn-NNew:],1)
            ZDat[i][lnn-NNew:]=(ZDat[i][lnn-NNew:]-P[1])/P[0]                      
    bbbbb=ZDat[:,:].transpose().copy()
    aaaaa=np.median(bbbbb.transpose(),axis=0)
    %varexp --plot bbbbb 
    len(arrrxx)
except:
    len(arrrxx)
len(arrrxx)

dNIt=8
Ngroup=3
Lo=1
Nproc=3*Ngroup#*(os.cpu_count())
wrkdir = r"C:/Work/WX16/"
[hhhao,Arr_AAA]=(hkl.load(wrkdir + "BNT-USD"+".rlf1"))
NIter=100

for hhhai in range(hhhao):  
    hhha=hhhai+1
    for iGr in range(Ngroup):  
        ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhha)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhha)*int(Nproc/Ngroup)].copy()
        if iGr==0:
            xxxx=ZDat.copy()
        else:
            xxxx=np.concatenate((xxxx, ZDat))
    ZDat=xxxx.copy()        
    
    anI=len(ZDat)
    for ii in range(3):       
        ar0x=np.median(ZDat,axis=0)
        ar0x_=.4*(np.median(abs((ZDat)-(ar0x)),axis=0))
            
        lnn=len(ZDat[0])
        NNew=int(.35*lnn)
        for i in range(anI):    
            for j in range(lnn):    
                if not abs(ZDat[i,j]-(ar0x[j]))<=ar0x_[j]:     
                    if ZDat[i,j]<((ar0x[j])-ar0x_[j]):            
                        ZDat[i,j]=(ar0x[j])-ar0x_[j]
                    else:
                        if ZDat[i,j]>((ar0x[j])+ar0x_[j]):
                            ZDat[i,j]=(ar0x[j])+ar0x_[j]
        anI=anI
        P=np.zeros(3,float)
        for i in range(anI):
            dd=ZDat[i][lnn-NNew:].copy()                         
            x=ar0x[lnn-NNew:lnn-NNew+int(lSrez*(NNew-(lnn-len(ar0x))))].copy()
            ZDat[i][lnn-NNew:]=filterFourierQ(ZDat[i],(ar0x),NNew,1)[lnn-NNew:]
            P[0:2]=np.polyfit(x,ZDat[i][lnn-NNew:lnn-NNew+int(lSrez*(NNew-(lnn-len(ar0x))))],1)
            if not P[0]>0:
                P[0:2]=np.polyfit(dd,ZDat[i][lnn-NNew:],1)
            ZDat[i][lnn-NNew:]=(ZDat[i][lnn-NNew:]-P[1])/P[0]                      
    #ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
    bbbbb=ZDat[:,:].transpose().copy()
    aaaaa=np.median(bbbbb.transpose(),axis=0)
    %varexp --plot bbbbb 

