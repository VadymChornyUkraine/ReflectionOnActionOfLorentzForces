import hickle as hkl
import numpy as np
arrrxx=hkl.load("ralfrez.rlf2")
dNIt=8
Ngroup=3
Lo=1
Nproc=2*Ngroup#*(os.cpu_count())
wrkdir = r"/home/vacho/Документи/Work/W14_7/WX11/"
[hhha,Arr_AAA]=(hkl.load(wrkdir + "ATOM-USD"+".rlf1"))
NIter=100

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
    ar0x_=1.4*np.median(abs(ZDat-np.log(ar0x)),axis=0)
else:
    ar0x=np.median(ZDat,axis=0)
    ar0x_=1.4*(np.median(abs((ZDat)-(ar0x)),axis=0))
    
for i in range(anI):    
    if Lo:                                
        ZDat[i]=(ZDat[i]*(abs(ZDat[i]-np.log(ar0x))<=ar0x_))+np.log(ar0x)*(abs(ZDat[i]-np.log(ar0x))>ar0x_)
    else:
        ZDat[i]=(ZDat[i]*(abs((ZDat[i])-(ar0x))<=ar0x_))+ar0x*(abs((ZDat[i])-(ar0x))>ar0x_)

#ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
bbbbb=ZDat[:,:].transpose().copy()
aaaaa=np.median(bbbbb.transpose(),axis=0)
%varexp --plot bbbbb

