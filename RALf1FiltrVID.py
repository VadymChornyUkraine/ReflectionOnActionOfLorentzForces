import numpy as np
from operator import itemgetter
import time as tm
import RALF1FilterX as XFilter
import sys
import lfib1340 
from scipy import stats as scp
#import win32api,win32process,win32con 
from random import sample 
#from scipy.signal import savgol_filter
#import wmi
import hickle as hkl
MaxTemp=82


def CheckTemp(aTemp=MaxTemp):
    # w = wmi.WMI(namespace="OpenHardwareMonitor")
    # temperature_infos = w.Sensor()
    # aMxTemp=0
    # for sensor in temperature_infos:
    #     if sensor.SensorType==u'Temperature':
    #         aMxTemp=np.maximum(sensor.Value,aMxTemp)
            
    # if aMxTemp>aTemp:
    #     return 0
    # else:
        return 1
        
# priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
#                win32process.BELOW_NORMAL_PRIORITY_CLASS,
#                win32process.NORMAL_PRIORITY_CLASS,
#                win32process.ABOVE_NORMAL_PRIORITY_CLASS,
#                win32process.HIGH_PRIORITY_CLASS,
#                win32process.REALTIME_PRIORITY_CLASS]  

DETERM=0.1

def RandomQ(Nfx,NQRandm_=0):
    while not CheckTemp:
        tm.sleep(60)
    QRandm_=np.asarray(range(512),float)
    NQRandm=0  
    KK=3e6
    liiX=np.zeros(Nfx,float)
    pp=0
    while pp<0.7:
        for ii in range(3):
            try:                
                z=(QRandm_[NQRandm]+1)/KK           
                atim0=tm.time()        
                tm.sleep(z) 
                atim=tm.time()     
                dd=int(((atim-atim0)/z-1)/1000)
                zz=np.asarray(sample(list(range(Nfx)),Nfx),float)/KK
                lfib1340.LFib1340(dd).shuffle(zz)  
                lfib1340.LFib1340(int(2*dd/(1+np.sqrt(5)))).shuffle(QRandm_)
                
                if NQRandm>0:
                    liiX=liiX+zz
                NQRandm=NQRandm+1
            except:
                NQRandm=0                

        k2, pp = scp.skewtest(liiX)
            
    rr2=[[],[]]
    rr2[0]= liiX.copy()
    rr2[1]= np.asarray(np.linspace(0,Nfx-1,Nfx),int)
    m=[[rr2[j][l] for j in range(len(rr2))] for l in range(len(rr2[0]))]         
    m.sort(key=itemgetter(0))                  
    rr2=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]  
    liiXX=np.asarray(rr2[1],int)
    return liiXX

def filterFourierQ(arxx,arb,NNew,NChan,key=1):  
    Nfl=int(len(arb)/NChan)
    Nfl_=int(len(arxx)/NChan)
    Nnl=NNew    
    
    ar_=np.zeros(Nnl,float)
    farx=np.zeros(2*Nnl,float)-1e32
    
    az=int(np.floor(Nfl/Nnl))
    
    gg0=0    
    for l in range(NChan):        
        for i in range(az):
            ar_=arb[Nfl-(az-i)*Nnl+Nfl*l:Nfl+Nnl-(az-i)*Nnl+Nfl*l].copy()
            gg0=gg0+np.sum(ar_*ar_)
            ar_=ar_-ar_[len(ar_)-1]
            ar__=abs(np.fft.fft(np.concatenate((ar_,-2*((key<0)-.5)*ar_)))) 
            farx=np.maximum(farx,ar__)
    gg0=np.sqrt(gg0)/(NChan*az*Nnl)
     
    gg=0
    arxr=arxx.copy()
    for l in range(NChan):      
        ar_=arxx[Nfl_*(l+1)-Nnl:Nfl_*(l+1)].copy()
        ar_=ar_-ar_[0]
        ar_=ar_[::-1].copy()
        farxx=np.fft.fft(np.concatenate((ar_,-2*((key<0)-.5)*ar_)))    
        mfarxx=np.abs(farxx)+1e-32  
        farxxx=np.zeros(2*Nnl,complex)    

        A1=np.exp(DETERM*np.std(np.log(mfarxx[3:2*Nnl-2]))+np.mean(np.log(mfarxx[3:2*Nnl-2])))
 
        for j in range(1,2*Nnl):
            if mfarxx[j]>A1:
                farxxx[j]=(farxx[j]/mfarxx[j])*np.sqrt(farx[j]*mfarxx[j]) 
                #farxxx[j]=(farxx[j]/mfarxx[j])*farx[j]
                       
        farxxx[0]=0*farxx[0]   
        if not key<0:
            farxxx[1]=farxxx[1]*0
            farxxx[2*Nnl-1]=farxxx[2*Nnl-1]*0 
        # if key<0:
        #     farxxx[2]=farxxx[2]*0
        #     farxxx[2*Nnl-2]=farxxx[2*Nnl-2]*0            
                   
        aaa=np.fft.ifft(farxxx).real
        aaa=(-2*((key<0)-.5)*aaa[0:int(len(aaa)/2)]+aaa[int(len(aaa)/2):])/2
        arxr[Nfl_-Nnl+Nfl_*l:Nfl_+Nfl_*l]=aaa[::-1].copy()[0:Nnl]
        gg=gg-arxr[Nfl_-Nnl+Nfl_*l]+arb[Nfl_-Nnl+Nfl_*l-1]

    gg=gg/NChan   
    for l in range(NChan):  
        arxr[Nfl_*l:Nfl_-Nnl+Nfl_*l]=arb[Nfl_*l:Nfl_-Nnl+Nfl_*l].copy()
        arxr[Nfl_-Nnl+Nfl_*l:Nfl_+Nfl_*l]=arxr[Nfl_-Nnl+Nfl_*l:Nfl_+Nfl_*l]+gg
    
    return arxr

def RALF1FilterQ(dQ2):    
    Np=len(dQ2)
    Nf=len(dQ2[0])
       
    SdQ=np.mean(dQ2,0)  
    sSdQ=np.std(np.asarray(SdQ,float))
    for i in range(Np):
        SdQj_ = np.std(np.asarray(dQ2[i] - SdQ,float))
        SdQj__ = np.std(np.asarray(dQ2[i],float))            
        if SdQj__ >0. and sSdQ>0.:
            dQ2[i] = np.asarray(dQ2[i] +SdQ * ((SdQj_ - sSdQ)/ sSdQ ),float)
        else:
            dQ2[i]=np.zeros(Nf,float)        
    return dQ2
 
import warnings

def RALF1Calculation(arr_bx,arr_c,Nf,NNew,NNew0,NChan,Nhh,iProc,Nproc):
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    sz=Nf*NChan
    MM=2
    Nzz=3
    Nhh=4
    
    Ndel=MM
    NCh=int(np.ceil(sz/Ndel))  
    Ndel0=MM
    NCh0=int(np.ceil(sz/Ndel0))   
          
    arr_b=np.asarray(arr_bx,float)
    
    hh=0  
    hh_=0      
    Nzz0=Nzz
    Atim_0=tm.time()  
    dd1_x=[]
    dd2_x=[]  
    while hh<Nhh:  
        if hh==0:  
            AMX=np.zeros((Nhh,sz),float)
            AMN=np.zeros((Nhh,sz),float)
            dd10=np.zeros((Nhh+1,sz),float)
            dd20=np.zeros((Nhh+1,sz),float)
            max_dd1=np.zeros((Nhh+1,sz),float)-1e32
            min_dd2=np.zeros((Nhh+1,sz),float)+1e32            
            arr_bbbxxx1=np.zeros((Nhh,sz),float)
            arr_bbbxxx2=np.zeros((Nhh,sz),float)

            Nzz=Nzz0
            rr2=np.zeros((Nhh+1,sz),float)
            rr2[hh]=np.asarray(arr_b,float)
        if hh==1:
            Nzz=int(Nzz0/2+1)
            
        try:
            arrrxx=hkl.load("ralfrez.rlf2")
        except:
            arrrxx=[]
        if len(arrrxx)>=Nproc: 
            return rr2[hh]/0 
        
        liix=np.zeros((sz,sz),int) 
        dQ3=np.zeros((sz,sz),float)
        mDD=np.zeros((sz,sz),float)  
        
        aa=RandomQ(sz)  
        liiB=np.concatenate((aa,aa,aa))  
        aa=RandomQ(sz) 
        liiD=np.concatenate((aa,aa,aa))          

        rrr=rr2[hh].copy()
            
        for i in range(sz):    
            liix[i]=liiB[liiD[i]:sz+liiD[i]].copy()
            dQ3[i]=rrr[liix[i]].copy()
        
        astart=np.Inf
        dQ3=dQ3.reshape((sz*sz))
        astart=dQ3[0]            
        dQ3[1:]=np.diff(dQ3)
        dQ3[0]=0
        dQ3=dQ3.reshape((sz,sz))
        dQ3_=dQ3.copy()
        
        D=np.std(dQ3)
        
        R4=np.ones(sz,float)  
        for l in range(NChan):
            R4[Nf-NNew+Nf*l:Nf+Nf*l]=0
            
        for i in range(sz):     
            mDD[i]=R4[liix[i]].copy() 
        
        seqq=(dQ3.reshape(sz*sz))[1:]*np.ceil(0.5*(1/(mDD.reshape(sz*sz)==1)[0:sz*sz-1]+1/(mDD.reshape(sz*sz)==1)[1:]))
        seqq_=  mDD.reshape(sz*sz)*0
        seqq_[0]=1
        seqq_[1:]=(abs(seqq)==np.Inf)+np.isnan(seqq)
        seqq_=seqq_.reshape((sz,sz))
        seqq=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqq)),float) 
        seqq=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqq)),float)

        ##########################################       
        aa=RandomQ(sz) 
        NumFri0=np.concatenate((aa, aa, aa))  
        aa=RandomQ(sz) 
        NumFri0_=np.concatenate((aa, aa, aa)) 
        aa=RandomQ(sz) 
        rR0=np.concatenate((aa, aa, aa))   
        aa=RandomQ(sz) 
        liiC=np.concatenate((aa, aa, aa)) 
        aa=RandomQ(sz)  
        r5=aa.copy()
        r5=r5*D/np.std(r5)
        r5=r5-np.mean(r5)
        r5=np.concatenate((r5, r5))
        aa=RandomQ(sz) 
        ss4=np.concatenate((aa, aa, aa, aa))                         
        zz=0  
        WW=0   

        while zz<Nzz and WW>-2*Nhh:  
            dQ3mx=np.zeros((sz,sz),np.float16)-1e32
            dQ3mn=np.zeros((sz,sz),np.float16)+1e32
            dQ3num=np.zeros((sz,sz),int)
            NumFri=NumFri0_[NumFri0[ss4[zz]]:NumFri0[ss4[zz]]+2*sz].copy()
            NumFri_=NumFri0[NumFri0_[ss4[zz]]:NumFri0_[ss4[zz]]+2*sz].copy()
            for kkk in range(Nzz):
                #dQ3=dQ3_.copy()
                aa=ss4[0]
                ss4[0:len(ss4)-1]=ss4[1:].copy()
                ss4[len(ss4)-1]=aa
                rR=rR0[liiC[ss4[zz]]:liiC[ss4[zz]]+2*sz].copy()
                rR_=rR0[liiC[ss4[len(ss4)-2*sz+zz]]:liiC[ss4[len(ss4)-2*sz+zz]]+2*sz].copy()
                
                kk=-1
                xxx=0
                while kk <Ndel-1 and xxx==0:  
                    kk=kk+1                      
                    ii=int(kk*NCh)
                    k=-1
                    while k<Ndel0-1 and xxx==0:     
                        k=k+1                       
                        i=int(k*NCh0) 
                        dQ4=np.zeros((NCh,NCh0),float)
                        mDD4=np.zeros((NCh,NCh0),float)
                        mDD4_=np.zeros((NCh,NCh0),float)
                        mDD4_A=np.zeros((NCh,NCh0),float) 
                        mDD4_B=np.zeros((NCh,NCh0),float)                                    
                        for ll in range(NCh0):
                            dQ4[:,ll]=dQ3[NumFri[ii:ii+NCh],NumFri_[i+ll]].copy()
                            mDD4[:,ll]=mDD[NumFri[ii:ii+NCh],NumFri_[i+ll]].copy()
                            mDD4_[:,ll]=r5[rR_[ss4[ll]+zz]:rR_[ss4[ll]+zz]+NCh].copy()
                        
                        mDD4_=(mDD4_-np.mean(mDD4_))           
                        P=np.zeros(3,float)
                        
                        for ll in range(NCh0):
                            mDD4_A[:,ll]=r5[rR[ss4[ll]+zz]:rR[ss4[ll]+zz]+NCh].copy()
                            mDD4_B[:,ll]=r5[rR[ss4[ll]+zz]+NCh:rR[ss4[ll]+zz]:-1].copy()
                       
                        mDD4_A=mDD4_A-np.mean(mDD4_A)
                        mDD4_B=mDD4_B-np.mean(mDD4_B)
                        
                        nNxA=sum(sum(mDD4==1))     
                        nNxA_=sum(sum(mDD4==0))  
                        # if nNxA>nNxA_ and nNxA_>0:  
                        if nNxA_>0:  
                            seqA=(dQ4.reshape(NCh*NCh0))[1:]*np.ceil(0.5*(1/(mDD4.reshape(NCh*NCh0)==1)[0:NCh*NCh0-1]+1/(mDD4.reshape(NCh*NCh0)==1)[1:]))
                            seqA_=  mDD4.reshape(NCh*NCh0)*0
                            seqA_[0]=1
                            seqA_[1:]=(abs(seqA)==np.Inf)+np.isnan(seqA)
                            seqA_=seqA_.reshape((NCh,NCh0))
                            seqA=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqA)),float) 
                            seqA=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqA)),float)

                            dQ4_A= np.asarray( XFilter.RALF1FilterX( (dQ4)*(1-seqA_)+seqA_*( dQ4-(mDD4_A)),len(dQ4),len(dQ4[0]),1,0)+seqA_*((mDD4_)),np.float16)
                            dQ4_B=-np.asarray( XFilter.RALF1FilterX(-(dQ4)*(1-seqA_)+seqA_*(-dQ4-(mDD4_B)),len(dQ4),len(dQ4[0]),1,0)+seqA_*((mDD4_)),np.float16)
                         
                            #dQ4=dQ4_B*(dQ4_B>0)*((dQ4_A+dQ4_B)>0)+dQ4_A*(dQ4_A<0)*((dQ4_A+dQ4_B)<0)#                      
                            dQ4=(dQ4_A+dQ4_B)/2
                            dQ4_A=dQ4.copy()
                            dQ4_B=dQ4.copy() 
                            
                            seqB=(dQ4.reshape(NCh*NCh0))[1:]*np.ceil(0.5*(1/(mDD4.reshape(NCh*NCh0)==1)[0:NCh*NCh0-1]+1/(mDD4.reshape(NCh*NCh0)==1)[1:]))
                            seqB=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqB)),float) 
                            seqB=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqB)),float)
                            try:       
                                P[0:2]=np.polyfit(seqA,seqB,1)
                                # P[0]=np.std(seqB)/np.std(seqA)
                                # P[1]=np.mean(seqB)-P[0]*np.mean(seqA)  
                                # P[2]=0
                                if 100*scp.pearsonr(seqA,seqB)[0]>50 and not abs(P[0]-1)>0.5:   
                                    dQ4_A=(dQ4_A-P[1])/P[0]
                                    dQ4_B=(dQ4_B-P[1])/P[0]                                      
                                    for ll in range(NCh0):
                                        dQ3mx[NumFri[ii:ii+NCh],NumFri_[i+ll]]=(#dQ3mx[NumFri[ii:ii+NCh],NumFri_[i+ll]]*dQ3num[NumFri[ii:ii+NCh],NumFri_[i+ll]]+
                                                                                np.maximum(dQ3mx[NumFri[ii:ii+NCh],NumFri_[i+ll]],dQ4_A[:,ll]))#/(dQ3num[NumFri[ii:ii+NCh],NumFri_[i+ll]]+1)
                                        dQ3mn[NumFri[ii:ii+NCh],NumFri_[i+ll]]=(#dQ3mn[NumFri[ii:ii+NCh],NumFri_[i+ll]]*dQ3num[NumFri[ii:ii+NCh],NumFri_[i+ll]]+
                                                                                np.minimum(dQ3mn[NumFri[ii:ii+NCh],NumFri_[i+ll]],dQ4_B[:,ll]))#/(dQ3num[NumFri[ii:ii+NCh],NumFri_[i+ll]]+1)
                                        dQ3num[NumFri[ii:ii+NCh],NumFri_[i+ll]]=dQ3num[NumFri[ii:ii+NCh],NumFri_[i+ll]]+1
                                        dQ3[NumFri[ii:ii+NCh],NumFri_[i+ll]]=(1-seqA_[:,ll])*dQ3_[NumFri[ii:ii+NCh],NumFri_[i+ll]]+seqA_[:,ll]*((dQ3mx+dQ3mn)[NumFri[ii:ii+NCh],NumFri_[i+ll]])/2
                                    ll=ll+1
                                else:
                                    xxx=1
                            except:
                                xxx=1
                        else:     
                            xxx=1
                        
            if xxx==0:    
                if zz==0:
                    AsrXMx=dQ3mx.copy()
                    AsrXMn=dQ3mn.copy()      
                else:  
                    # AsrXMx=(AsrXMx*zz+dQ3mx)/(zz+1)
                    # AsrXMn=(AsrXMn*zz+dQ3mn)/(zz+1)
                    AsrXMx=np.maximum(AsrXMx,dQ3mx)
                    AsrXMn=np.minimum(AsrXMn,dQ3mn)

                AsrXMx_=AsrXMx.copy()
                AsrXMn_=AsrXMn.copy()
                
                dQ3=dQ3_.copy()
                WW=0                                    
                zz=zz+1
            else:
                aa=ss4[0]
                ss4[0:len(ss4)-1]=ss4[1:].copy()
                ss4[len(ss4)-1]=aa
                WW=WW-1  
        
        try:
            arrrxx=hkl.load("ralfrez.rlf2")
        except:
            arrrxx=[]
        if len(arrrxx)>=Nproc: 
            return rr2[hh]/0 
        
        hh0=hh
        if not WW<0:   
            seqq_=(( (AsrXMx_+AsrXMn_)/2  ).reshape(sz*sz))[1:]*np.ceil(0.5*(1/(mDD.reshape(sz*sz)==1)[0:sz*sz-1]+1/(mDD.reshape(sz*sz)==1)[1:]))
            seqq_=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqq_)),float) 
            seqq_=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqq_)),float)
            P[0:2]=np.polyfit(seqq,seqq_,1)
            if not abs(P[0]-1)>0.5 and 100*scp.pearsonr(seqq,seqq_)[0]>50:
                AsrXMx_=(AsrXMx_-P[1])/P[0] 
                AsrXMn_=(AsrXMn_-P[1])/P[0] 
                
            if not astart==np.Inf:
                dd1=AsrXMx_.reshape((sz*sz))
                dd1=astart+np.cumsum(dd1)
                AsrXMx_=dd1.reshape((sz,sz))
                dd1=AsrXMn_.reshape((sz*sz))
                dd1=astart+np.cumsum(dd1)
                AsrXMn_=dd1.reshape((sz,sz))
                
            dQ3=(AsrXMx_+AsrXMn_)/2    
            AsrXMx_=dQ3.copy()
            AsrXMn_=dQ3.copy()
            
            aMx_=0
            aMn_=0
            aMx=np.zeros(sz,float)-1e32
            aMn=np.zeros(sz,float)+1e32
            for i in  range(sz):
                aMx[liix[i]]=np.maximum(aMx[liix[i]],AsrXMx_[i])
                aMn[liix[i]]=np.minimum(aMn[liix[i]],AsrXMn_[i])
                aMx_=(aMx_*i+aMx)/(i+1)#
                aMn_=(aMn_*i+aMn)/(i+1)#
                # aMx_=aMx
                    
                # aMn_=aMn
            ann=sum(np.isnan(aMx_ + aMn_))
            if ann==0: 
                if hh==0: 
                    AMX[hh]=aMx_.copy()
                    AMN[hh]=aMn_.copy()  
                    arr_bbbxxx1[hh]=AMX[hh].copy()
                    arr_bbbxxx2[hh]=AMN[hh].copy()
                    D1=10
                else:
                    AMX[hh]=aMx_.copy()
                    AMN[hh]=aMn_.copy()
                    arr_bbbxxx1[hh]=(arr_bbbxxx1[hh-1]*(hh-1)+AMX[hh])/(hh)
                    arr_bbbxxx2[hh]=(arr_bbbxxx2[hh-1]*(hh-1)+AMN[hh])/(hh)  
                    dd1_=[]
                    dd2_=[]                    
                    D1_=[]
                    D2_=[]
                    for l in range(NChan):
                        dd1_.append(arr_bbbxxx1[hh-1][Nf-NNew+Nf*l:Nf+Nf*l].copy())  
                        dd2_.append(arr_bbbxxx2[hh-1][Nf-NNew+Nf*l:Nf+Nf*l].copy())  
                        D1_.append((arr_bbbxxx1[hh]-arr_bbbxxx1[hh-1])[Nf-NNew+Nf*l:Nf+Nf*l].copy())
                        D2_.append((arr_bbbxxx2[hh]-arr_bbbxxx2[hh-1])[Nf-NNew+Nf*l:Nf+Nf*l].copy())
                    D1_=np.asarray(D1_,float)
                    D2_=np.asarray(D2_,float)
                    dd1_=np.asarray(dd1_,float)
                    dd2_=np.asarray(dd2_,float)                  
                    D1=np.std(D1_+D2_)/np.std(dd1_+dd2_)*np.sqrt(hh+1)/np.sqrt(1+(hh+1))  
                    
                ann=1  
                try:
                    dd1=(filterFourierQ(AMX[hh],rrr,NNew,NChan))
                    dd2=(filterFourierQ(AMN[hh],rrr,NNew,NChan)) 
                    if sum(np.abs(dd1+dd2)==np.Inf)==0 and D1>DETERM:  
                        dd1_x.append(dd1)
                        dd2_x.append(dd2)
                        dd1=np.amax(dd1_x,axis=0)
                        dd2=np.amin(dd2_x,axis=0)
                        sr2_1=[]
                        sr2_2=[]
                        sarr_c=[]
                        for l in range(NChan):  
                            sr2_1.append((dd1)[Nf-NNew+Nf*l:Nf-NNew0+Nf*l])
                            sr2_2.append((dd2)[Nf-NNew+Nf*l:Nf-NNew0+Nf*l])
                            sarr_c.append(arr_c[(NNew-NNew0)*l:NNew-NNew0+(NNew-NNew0)*l].copy())
                        sr2_1=np.asarray(sr2_1,float)
                        sr2_2=np.asarray(sr2_2,float)
                        sarr_c=np.asarray(sarr_c,float) 
                        sr2_1=sr2_1.reshape((len(sr2_1)*len(sr2_1[0])))
                        sr2_2=sr2_2.reshape((len(sr2_2)*len(sr2_2[0])))
                        sarr_c=sarr_c.reshape((len(sarr_c)*len(sarr_c[0])))   
                    
                        P_1=P.copy()
                        P_2=P.copy()
                        # P_1[0:2]=np.polyfit(sarr_c,sr2_1,1)
                        # P_2[0:2]=np.polyfit(sarr_c,sr2_2,1)
                        
                        P_1[0]=np.std(sr2_1)/np.std(sarr_c)
                        P_1[1]=np.mean(sr2_1)-P_1[0]*np.mean(sarr_c) 
                        P_2[0]=np.std(sr2_2)/np.std(sarr_c)
                        P_2[1]=np.mean(sr2_2)-P_2[0]*np.mean(sarr_c)  
                        if P_1[0]>0 and P_2[0]>0:
                        # if 100*scp.pearsonr(sarr_c,((sr2_1-P_1[1])/P_1[0]+
                        #     (sr2_2-P_2[1])/P_2[0]))>10:                        
                            for l in range(NChan):  
                                dd1[Nf-NNew+Nf*l:Nf+Nf*l]=(dd1[Nf-NNew+Nf*l:Nf+Nf*l]-P_1[1])/P_1[0]
                                dd2[Nf-NNew+Nf*l:Nf+Nf*l]=(dd2[Nf-NNew+Nf*l:Nf+Nf*l]-P_2[1])/P_2[0]
                            max_dd1[hh]=rr2[hh].copy()
                            min_dd2[hh]=rr2[hh].copy()
                            for l in range(NChan):
                                if hh==0:
                                     max_dd1[hh,Nf-NNew+Nf*l:Nf+Nf*l]=dd1[Nf-NNew+Nf*l:Nf+Nf*l].copy()
                                     min_dd2[hh,Nf-NNew+Nf*l:Nf+Nf*l]=dd2[Nf-NNew+Nf*l:Nf+Nf*l].copy()
                                else:
                                     max_dd1[hh,Nf-NNew+Nf*l:Nf+Nf*l]=(max_dd1[hh-1,Nf-NNew+Nf*l:Nf+Nf*l]*hh+
                                                                       (dd1[Nf-NNew+Nf*l:Nf+Nf*l]))/(hh+1)
                                     min_dd2[hh,Nf-NNew+Nf*l:Nf+Nf*l]=(min_dd2[hh-1,Nf-NNew+Nf*l:Nf+Nf*l]*hh+
                                                                       (dd2[Nf-NNew+Nf*l:Nf+Nf*l]))/(hh+1)
                                                                               
                            hh=hh+1
                            ann=0         
                            rr2[hh]=(max_dd1[hh-1]+min_dd2[hh-1])/2
                            rr2[hh]=filterFourierQ(rr2[hh],rr2[hh-1],NNew,NChan) 
                            sr2=[]
                            sarr_c=[]
                            for l in range(NChan):  
                                sr2.append(rr2[hh,Nf-NNew+Nf*l:Nf-NNew0+Nf*l].copy())
                                sarr_c.append(arr_c[(NNew-NNew0)*l:NNew-NNew0+(NNew-NNew0)*l].copy())
                            sr2=np.asarray(sr2,float)
                            sarr_c=np.asarray(sarr_c,float) 
                            sr2=sr2.reshape((len(sr2)*len(sr2[0])))
                            sarr_c=sarr_c.reshape((len(sarr_c)*len(sarr_c[0])))   
                            
                            P[0:2]=np.polyfit(sarr_c,sr2,1)  
                            
                            if P[0]>0:  
                                #100*scp.pearsonr(sarr_c,sr2)[0]>10 and
                                dd1_x=[]     
                                dd2_x=[]  
                                for l in range(NChan):  
                                    rr2[hh,Nf-NNew+Nf*l:Nf+Nf*l]=(rr2[hh,Nf-NNew+Nf*l:Nf+Nf*l]-P[1])/P[0]
                                    rr2[hh,Nf*l:Nf-NNew+Nf*l]=arr_b[Nf*l:Nf-NNew+Nf*l].copy()
                                #P[0:2]=np.polyfit(sarr_c,sr2,1)   
    
                                #D1=0.2
                                if hh==Nhh:
                                    # sr2=[]
                                    # sarr_c=[]
                                    # for l in range(NChan):  
                                    #     sr2.append(rr2[hh,Nf-NNew+Nf*l:Nf-NNew0+Nf*l].copy())
                                    #     sarr_c.append(arr_c[(NNew-NNew0)*l:NNew-NNew0+(NNew-NNew0)*l].copy())
                                    # sr2=np.asarray(sr2,float)
                                    # sarr_c=np.asarray(sarr_c,float) 
                                    # sr2=sr2.reshape((len(sr2)*len(sr2[0])))
                                    # sarr_c=sarr_c.reshape((len(sarr_c)*len(sarr_c[0]))) 
                                    # P[0]=np.std(sr2)/np.std(sarr_c)
                                    # P[1]=np.mean(sr2)-P[0]*np.mean(sarr_c)
                                    if abs(P[0]-1.)<0.5 and sum(abs(rr2[hh])==np.Inf)==0 and D1<1: 
                                        anamef="fralf.tmp"
                                        fo = open(anamef, "w")
                                        Atim_1=tm.time()   
                                        fo.write(str(iProc)+' it=%d, K=%3.2f Tm=%5.2f\n'%(hh,D1,Atim_1-Atim_0))
                                        fo.close()
                                        return np.asarray(rr2[hh],float)
                                    else:
                                        hh=hh0
                            else:                                
                                hh=hh0
                        else:
                            hh=hh0
                except:
                    hh=hh0

        if hh0==hh:            
            hh_=hh_+1
            if hh>1:
                hh=hh-2
            else:
                hh=0
                
            if hh_>2*Nhh:
                return rr2[hh]/0                                                 

def RALf1FiltrQ(args):  
    # pid = win32api.GetCurrentProcessId()
    # handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    # win32process.SetPriorityClass(handle, priorityclasses[1])
    NChan=int(args[1])
    NNew=int(args[2])
    Nhh=int(args[3])
    Nf=int(len(args)-5) 
    Nproc=int(args[int(len(args)-1)])
       
    arr_bb=[]    
    for i in range(Nf):
        arr_bb.append(args[4+i])
    arr_bb=np.asarray(arr_bb,float)
    Nf=int(len(arr_bb)/NChan)    
    NNew0=int(NNew*1.2) 
    
    #Nhh=1
    Nhh0=Nhh
    astar0=np.Inf
    # astar0=arr_bb[0]
    # arr_bb[1:]=np.diff(arr_bb)
    # arr_bb[0]=0
    
    arr_b=np.asarray(arr_bb,float)
    
    arr_c=[]
    aa=RandomQ(NNew0)
    aa=aa-np.mean(aa)
    aa=2*aa/np.std(aa)*np.std(arr_b)
    ss4=np.concatenate((aa, aa, aa, aa)) 
    for l in range(NChan):
        arr_c.append(arr_b[Nf-NNew0+Nf*l:Nf-NNew+Nf*l].copy()) 
        arr_b[Nf-NNew0-1+Nf*l:Nf+Nf*l]=(astar0==np.Inf)*arr_b[Nf-NNew0+Nf*l-1]+ss4[l:NNew0+l+1]
    arr_c=np.asarray(arr_c,float)
    arr_c=arr_c.reshape((len(arr_c)*len(arr_c[0])))
    
    while 1==1: 
        hh=0
        ann=0
        arr_bbx=np.zeros((Nhh*2,Nf),float)
        Nch=0
        Koef=np.zeros(Nhh0+1,float)                
        KoefA=np.zeros(Nhh0+1,float)
        NumIt=3*(int((Nhh0)/2)+1)

        while hh<Nhh:
            if hh<Nhh:    
                arr_bbbxxx=RALF1Calculation(arr_b,arr_c,Nf,NNew0,NNew,NChan,NumIt,args[0],Nproc)                
                try:
                    arrrxx=hkl.load("ralfrez.rlf2")
                except:
                    arrrxx=[]
                if len(arrrxx)>=Nproc: 
                    return arr_bbbxxx/0 
                if (sum(np.abs(arr_bbbxxx)==np.Inf)==0 and sum(np.isnan(arr_bbbxxx))==0):
                    Nf_=NNew0+int(NNew0*1.2)
                    if Nf_>=Nf:
                        Nf_=Nf-1
                    arr_bbbxxx_=np.zeros(Nf_*NChan,float)
                    arr_c_=[]
                    aa=RandomQ(NNew0)
                    aa=aa-np.mean(aa)
                    aa=2*aa/np.std(aa)*np.std(arr_bbbxxx)
                    ss4=np.concatenate((aa, aa, aa, aa)) 
                    for l in range(NChan):
                        arr_bbbxxx_[Nf_*l:Nf_+Nf_*l]=np.asarray(arr_bbbxxx[Nf*(l+1):Nf-Nf_-1+Nf*l:-1],float)
                        arr_c_.append(arr_bbbxxx_[Nf_-NNew0+Nf_*l:Nf_-NNew+Nf_*l].copy())
                        arr_bbbxxx_[Nf_-NNew0+Nf_*l:Nf_+Nf_*l]=(astar0==np.Inf)*arr_bbbxxx_[Nf_-NNew0+Nf_*l-1]+ss4[l:NNew0+l]                     
                    arr_c_=np.asarray(arr_c_,float)
                    arr_c_=arr_c_.reshape((len(arr_c_)*len(arr_c_[0]))) 
                    arr_bbbxxx_y=RALF1Calculation(arr_bbbxxx_,arr_c_,Nf_,NNew0,NNew,NChan,NumIt,args[0],Nproc)
                    if (sum(np.abs(arr_bbbxxx_y)==np.Inf)==0 and sum(np.isnan(arr_bbbxxx_y))==0): 
                        for l in range(NChan):
                            if not astar0==np.Inf:
                                if l==0:
                                    mm1=np.cumsum(arr_bb[Nf-NNew0-(Nf_-NNew0)+Nf*l:Nf*(l+1)-(Nf_-NNew0)])
                                    mm2=np.cumsum(arr_bbbxxx_y[Nf_*(l+1):Nf_-NNew0-1+Nf_*l:-1])
                                else:
                                    mm1=np.concatenate((mm1,np.cumsum(arr_bb[Nf-NNew0-(Nf_-NNew0)+Nf*l:Nf*(l+1)-(Nf_-NNew0)])))
                                    mm2=np.concatenate((mm2,np.cumsum(arr_bbbxxx_y[Nf_*(l+1):Nf_-NNew0-1+Nf_*l:-1])))                
                            else:
                                if l==0:
                                    mm1=arr_bb[Nf-NNew0-(Nf_-NNew0)+Nf*l:Nf*(l+1)-(Nf_-NNew0)].copy()
                                    mm2=arr_bbbxxx_y[Nf_*(l+1):Nf_-NNew0-1+Nf_*l:-1].copy()
                                else:
                                    mm1=np.concatenate((mm1,arr_bb[Nf-NNew0-(Nf_-NNew0)+Nf*l:Nf*(l+1)-(Nf_-NNew0)].copy()))
                                    mm2=np.concatenate((mm2,arr_bbbxxx_y[Nf_*(l+1):Nf_-NNew0-1+Nf_*l:-1].copy()))                

                        ann=(sum(np.abs(mm1)==np.Inf)>0 + sum(np.isnan(mm1))>0+
                              sum(np.abs(mm2)==np.Inf)>0 + sum(np.isnan(mm2))>0)
                        
                        if ann==0 and len(mm1)>1 and len(mm1)==len(mm2):                             
                            if np.std(mm1)>0 and np.std(mm2)>0:
                                coef=100*(scp.pearsonr(mm1,mm2)[0])
                                if coef>0:
                                    anamef="fralf_.tmp"
                                    fo = open(anamef, "w")
                                    fo.write(str(args[0])+' %s'%(coef)+'\n')
                                    fo.close() 
                                
                                KoefA[hh]=coef
         
                                #mm1=mm1*np.std(mm2)/np.std(mm1)                       
                                Koef[hh]=-np.std(mm1-mm2)
                                if not astar0==np.Inf:
                                    arr_bbx[hh]=astar0+np.cumsum(arr_bbbxxx)
                                else:
                                    arr_bbx[hh]=arr_bbbxxx.copy()
                                #hh=hh+1
                
            # if hh==Nhh:
                                
                                try:
                                    arrrxx=hkl.load("ralfrez.rlf2")
                                    if len(arrrxx)>=Nproc: 
                                        return np.asarray(arr_bbx[hh],float)
                                    else:
                                        print('%s'%(coef))
                                        if coef>0:   
                                            arrrxx.append(np.asarray(arr_bbx[hh],float))
                                            try:
                                                hkl.dump(arrrxx,"ralfrez.rlf2")
                                            except:
                                                tm.sleep(0.6)
                                                hkl.dump(arrrxx,"ralfrez.rlf2")
                                except:
                                    arrrxx=[]


                                            
                # arr_bbx_=np.asarray(arr_bbx[0:Nhh],float)
                # r2s=np.zeros((2,Nhh),float)
                # r2s[0]= np.asarray(Koef[0:Nhh],float)
                # r2s[1]= np.asarray(range(Nhh),float)
                # m=[[r2s[j][l] for j in range(len(r2s))] for l in range(len(r2s[0]))]         
                # m.sort(key=itemgetter(0))                  
                # r2s=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]  
                # Nch=int(r2s[1][Nhh-1])
                # if np.isnan(KoefA[Nch]):
                #     KoefA[Nch]=0            
                # if KoefA[Nch]>=0:
                #     print(KoefA[0:Nhh])
                #     REZ=arr_bbx_[Nch].copy()
                #     return REZ
                # else:
                #     if (Nhh<len(KoefA)):
                #         Nhh=Nhh+1
                #     else:
                #         hh=0
                #         Nhh=Nhh0


if __name__ == '__main__':
    RALf1FiltrQ(sys.argv)