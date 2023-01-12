import hickle as hkl
import numpy as np
arrrxx=hkl.load("ralfrez.rlf2")
dNIt=8
Ngroup=3
Nproc=2*Ngroup#*(os.cpu_count())
wrkdir = r"/home/vacho/Документи/Work/W14_7/WX4/"
[hhha,Arr_AAA]=(hkl.load(wrkdir + "LRC-USD"+".rlf1"))
hhh=0
iGr=1
NIter=100
ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
aaaaa=ZDat[:,:].transpose().copy()
bbbbb=np.median(aaaaa.transpose(),axis=0)
#%varexp --plot aaaaa
#filterFourierQ(aaaaa[:,0],aaaaa[:,0],50,1,1)

