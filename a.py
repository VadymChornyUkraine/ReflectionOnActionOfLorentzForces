import hickle as hkl
import numpy as np

aaaa=hkl.load("ralfrez.rlf1")

dNIt=8
Ngroup=3
Nproc=2*Ngroup#*(os.cpu_count())
wrkdir = r"/home/vacho/Документи/Work/W14_7/WWW/"
[hhha,Arr_AAA]=(hkl.load(wrkdir + "BTC-USD"+".rlf1"))
hhh=0
iGr=0
NIter=100
ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
aaaaa=ZDat[:,:].transpose().copy()
bbbbb=np.median(aaaaa.transpose(),axis=0)
#%varexp --plot aaaaa
#filterFourierQ(aaaaa[:,0],aaaaa[:,0],50,1,1)

