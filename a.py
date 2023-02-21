import hickle as hkl
import numpy as np
arrrxx=hkl.load("ralfrez.rlf2")
dNIt=8
Ngroup=3
Nproc=2*Ngroup#*(os.cpu_count())
wrkdir = r"/home/vacho/Документи/Work/W14_7/WX10/"
[hhha,Arr_AAA]=(hkl.load(wrkdir + "pi"+".rlf1"))
NIter=100

for iGr in range(Ngroup):  
    ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhha)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhha)*int(Nproc/Ngroup)].copy()
    if iGr==0:
        xxxx=ZDat.copy()
    else:
        xxxx=np.concatenate((xxxx, ZDat))
ZDat=xxxx.copy()        
#ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
aaaaa=ZDat[:,:].transpose().copy()
bbbbb=np.median(aaaaa.transpose(),axis=0)
#%varexp --plot aaaaa
#filterFourierQ(aaaaa[:,0],aaaaa[:,0],50,1,1)

