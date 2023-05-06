import hickle as hkl
import numpy as np
arrrxx=hkl.load("ralfrez.rlf2")
# i=0 
# arrrxx_=[]
# for ii in arrrxx:
#     if not i==4:
#         arrrxx_.append(ii)
#     i=i+1
#hkl.dump(arrrxx_,"ralfrez.rlf2")
bbb=np.asarray(arrrxx,float).transpose()
%varexp --plot bbb
len(arrrxx)

dNIt=8
Ngroup=3
Lo=1
Nproc=3*Ngroup#*(os.cpu_count())
wrkdir = r"C:/Users/VadymChornyy/Desktop/Work/WX15/"
[hhhao,Arr_AAA]=(hkl.load(wrkdir + "BTC-USD"+".rlf1"))
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
    # for i in range(anI):  
    #     if Lo:
    #         ZDat[i][:len(ar0)]=np.log(ar0)
    #     else:
    #         ZDat[i][:len(ar0)]=ar0.copy()
    
    if Lo:
        ar0x=np.exp(np.median(ZDat,axis=0))  
        ar0x_=.04*np.median(abs(ZDat-np.log(ar0x)),axis=0)
    else:
        ar0x=np.median(ZDat,axis=0)
        ar0x_=.04*(np.median(abs((ZDat)-(ar0x)),axis=0))
        
    lnn=len(ZDat[0])
    for i in range(anI):    
        for j in range(lnn):    
            if Lo:             
                if not abs(ZDat[i,j]-np.log(ar0x[j]))<=ar0x_[j]:     
                    if ZDat[i,j]<(np.log(ar0x[j])-ar0x_[j]):            
                        ZDat[i,j]=np.log(ar0x[j])-ar0x_[j]
                    else: 
                        if ZDat[i,j]>(np.log(ar0x[j])+ar0x_[j]):
                            ZDat[i,j]=np.log(ar0x[j])+ar0x_[j]
            else:
                if not abs(ZDat[i,j]-(ar0x[j]))<=ar0x_[j]:     
                    if ZDat[i,j]<((ar0x[j])-ar0x_[j]):            
                        ZDat[i,j]=(ar0x[j])-ar0x_[j]
                    else:
                        if ZDat[i,j]>((ar0x[j])+ar0x_[j]):
                            ZDat[i,j]=(ar0x[j])+ar0x_[j]
                    
    #ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
    bbbbb=ZDat[:,:].transpose().copy()
    aaaaa=np.median(bbbbb.transpose(),axis=0)
    %varexp --plot bbbbb 

