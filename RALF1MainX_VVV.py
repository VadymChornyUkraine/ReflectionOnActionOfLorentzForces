#import concurrent.futures
import pylab as plt
import numpy as np
import urllib.request, json 
import pandas as pd
from PIL import Image  
import cv2 as cv
#import time as tm 
import hickle as hkl
import os
import multiprocessing as mp
import scipy.interpolate as interp

from scipy import stats as scp
import dateutil.parser
from operator import itemgetter
import dill 
#from scipy.signal import savgol_filter
import wmi as wm

from RALf1FiltrVID import filterFourierQ
from RALf1FiltrVID import RALf1FiltrQ
from RALf1FiltrVID import RandomQ
import RALF1FilterX as XFilter      
ticker="LRC-EURY"
ticker1="LRC-EUR"
ticker2="LRC-EUR"
aKEY=0
aname=ticker

wrkdir = r"c:\Work\\W14_7\\"
api_key = 'ONKTYPV6TAMZK464' 
 
interv="15min"
interv="Daily"

#INTRADAY
#d_intervals = {"1min","5min","15min","30min","60min"}

Lengt=400
Ngroup=3
Nproc=2*Ngroup#*(os.cpu_count())
Lo=1  
lSrez=0.99
aTmStop=6
NIt=3
NIter=100
dNIt=8
DT=0.2
aDecm=2
KPP=0
    
def decimat(adat_):
    if aDecm<1:
        if Lo:            
            f=interp.interp1d(np.asarray(range(len(adat_)),float), np.log(adat_),'linear')
            xnew=np.asarray(range(int((len(adat_)-1)/aDecm)),float)*aDecm
            adat__=np.exp(np.asarray(f(xnew),float))
        else:
            f=interp.interp1d(np.asarray(range(len(adat_)),float), adat_,'linear')
            xnew=np.asarray(range(int((len(adat_)-1)/aDecm)),float)*aDecm
            adat__=np.asarray(f(xnew),float)
    else:
        if Lo:
            adat_=np.log(adat_)
        adatx=0
        k=0
        adat__=np.zeros(int(len(adat_)/aDecm),float)
        for i in range(int(len(adat_)/aDecm)):
            adat__[k]=np.mean(adat_[i*aDecm:i*aDecm+aDecm])
            k=k+1
        if Lo:
            return np.exp(adat__[0:len(adat__)])
    return (adat__[0:len(adat__)])

def fig2img ( fig ):
    fig.savefig(wrkdir +ticker+'dynamic.png',dpi=150,transparent=False,bbox_inches = 'tight')
    frame=Image.open(wrkdir +ticker+'dynamic.png')
    # fig.canvas.draw()
    # frame=Image.frombytes('RGB', fig.canvas.get_width_height(),
    #                        fig.canvas.tostring_rgb())
    return frame

def loaddata(aLengt,ticker1,key):
    adat_=[]
    url_string =  "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=%s&outputsize=full&apikey=%s"%(ticker1,interv,api_key)        
    if interv=="Daily":
        url_string =  "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker1,api_key)

    if key>0:  
        data = json.loads(urllib.request.urlopen(url_string).read().decode())['Time Series (%s)'%(interv)]
        df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
        arrr=[]
        adate=[]
        adt=[]
        for k,v in data.items():
            date = dateutil.parser.parse(k)
            data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
            df.loc[-1,:] = data_row
            if Lo:
                rr=np.sqrt(np.asarray(data_row)[1]*np.asarray(data_row)[2])
            else:
                rr=(np.asarray(data_row)[1]+np.asarray(data_row)[2])/2
            if rr!=0:
                adate.append(date.timestamp())
                adt.append(k)
                arrr.append(rr)
            df.index = df.index + 1
    #        if np.asarray(arrr,int).size>=aLengt:#495:1023:
    #            break
        aa=[[] for i in range(3)]
        aa[0]=adate
        aa[1]=arrr  
        aa[2]=adt
        m=aa
        aa=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]   
        aa.sort(key=itemgetter(0))
        m=aa
        aa=[[m[j][l] for j in range(len(m))] for l in range(len(m[0]))]     
        ada=list(aa)[2]
        arrr=list(aa)[1]
        sz=np.asarray(arrr).size
        ln=min(sz,aLengt)
        arr=np.asarray(arrr).copy()
        arrr=[]        
        for i in range(ln-1):
            arrr.append(arr[sz-ln+i])
            adat_.append(ada[sz-ln+i])
    else:
        file=open(wrkdir+ticker1+ '.txt','r')
        arrr=np.asarray(file.readlines(),float).copy()
        if len(arrr)>aLengt:
            arrr=arrr[len(arrr)-aLengt:]
        file.close()
        
    return arrr,adat_

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

if __name__ == '__main__': 
    w = wm.WMI(namespace="OpenHardwareMonitor")
    temperature_infos = w.Sensor()
    if temperature_infos==[]:
        os.startfile(r".\\OpenHardwareMonitor\OpenHardwareMonitor.exe")

    del(w)
    del(temperature_infos)
    del(wm)
    
    try:
        dill.load_session(wrkdir + aname+".ralf")
    except:
        ImApp=[]
        try:
            arrrxx=hkl.load(wrkdir + aname+"dat.rlf1")
        except:

            #ticker ="EOSOMG" # "BTCUSD"#"GLD"#"DJI","LOIL.L"#""BZ=F" "LNGA.MI" #"BTC-USD"#"USDUAH"#"LTC-USD"#"USDUAH"#
            if not ticker1==ticker2:
                arrrxx1,adat1_=loaddata(Lengt,ticker1,aKEY)
                arrrxx1=np.asarray(arrrxx1,float)
                arrrxx2,adat2_=loaddata(Lengt,ticker2,aKEY)
                arrrxx2=np.asarray(arrrxx2,float)
                lnm=min(len(arrrxx1),len(arrrxx2))
                arrrxx=arrrxx1[len(arrrxx1)-lnm:]/arrrxx2[len(arrrxx2)-lnm:]
            else:
                ticker=ticker1
                arrrxx,adat_=loaddata(Lengt,ticker,aKEY)
                arrrxx=np.asarray(arrrxx,float)
            
            arrrxx=decimat(arrrxx)
                
            try:                
                hkl.dump(arrrxx,wrkdir + aname+"dat.rlf1")
            except:
                os.mkdir(wrkdir)
                hkl.dump(arrrxx,wrkdir + aname+"dat.rlf1")
            
        esz=len(arrrxx)
        arr_rezDzRez=[[] for j in range(esz)]
        kkk=0
        out=0   

        aname=aname
        arrr=np.asarray(arrrxx).copy()  

        arrr=np.asarray(arrr,float)    
        Lengt=len(arrr)
        Nf=Lengt
        
        nn=int(Nf*DT)             
        NNew=int(Nf*0.4)  
        Nf=Nf+nn        
        ar0=np.asarray(arrr[0:])           
        
        arr_z=np.zeros(Nf,float)
        arr_z[0:Nf-NNew]=arrr[0:Nf-NNew].copy()
        arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]
          
        adat0=''
                               
        Arr_AAA=np.zeros((NIter*Nproc,Nf),float) 
        arr_rezBz=np.zeros(Nf,float)
        arr_rezBzz=arr_rezBz.copy()
        
        all_rezAz=np.zeros((NIter,Nf),float)
        arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]  
        all_RezN=np.zeros((Ngroup,NIter,Nf),float)+1e32
        all_RezM=np.zeros((Ngroup,NIter,Nf),float)-1e32
        dd1a=np.zeros((Ngroup,NIter,Nf),float)-1e32
        dd2a=np.zeros((Ngroup,NIter,Nf),float)+1e32
        all_RezNM=np.zeros((Ngroup,NIter,Nf),float)
        all_RezMM=np.zeros((Ngroup,NIter,Nf),float)
        argss=[[0] for j in range(Nproc)]    

        hh0=0
        hhh=0
        hhh_=0
                
        Koef_=[]
        ZZ=0
        key=0
        try:
            dill.load_session(wrkdir + aname+".ralf")
        except:    
            fig = plt.figure()
            axes = fig.add_axes([0.1, 0.1, 1.2, 1.2])
            axes_ = fig.add_axes([0, 0, 0.3, 0.3])     
            axes__ = fig.add_axes([0.4, 0, 0.3, 0.3])
            
            if Lo:
                axes.semilogy(ar0, 'r.')
                axes.semilogy(arr_z, 'go-')  #cut data used for model
                axes.grid(True, which="both", ls="-")
            else:
                axes.plot(ar0, 'r.')
                axes.plot(arr_z, 'go-')  #cut data used for model
                axes.grid(True, which="both", ls="-")

            axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s, start = %s, step = %s * %s'%(aname,adat0,interv,aDecm),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=axes_.transAxes,color='blue', fontsize=14)        

            try:
                frame=fig2img(fig)  
            except:
                frame=fig2img(fig) 
            ImApp.append(frame)
            cimg = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)        
            gray_sz1=len(cimg[0])
            gray_sz2=len(cimg)
            aDur=2
            fourcc = cv.VideoWriter_fourcc(*'MP4V')
            out = cv.VideoWriter(wrkdir + aname+'.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
            for icl in range(len(ImApp)):
                cimgx=(cv.cvtColor(np.array(ImApp[icl]), cv.COLOR_RGB2BGR)) 
                out.write(cimgx[0:gray_sz2,0:gray_sz1,:]) 
            out.release()
            plt.show()
            del(out)

    while hhh_<aTmStop and not key == 13: 
        Aprocess=[]
        if hhh==int(NIter/1):
            if hhh_<aTmStop-1:
                try:
                    os.remove(wrkdir + aname+".rlf1")
                except:
                    hhh_=hhh_
                nnn=int(nn*0.5)
                aTmStop=6
                hh0=0
                hhh=0
                arr_z=np.zeros(Nf+nnn,float)
                arr_z[0:nnn]=arr_rezBz[0:nnn].copy()
                arr_z[nnn:Nf]=arr_rezBz[nnn:Nf].copy()
                arr_z[Nf:]=arr_z[Nf-1]
                Lengt=Lengt+nnn
                ar0=arr_z[0:Lengt-nnn].copy()
                Nf=Nf+nnn
                arr_rezBz=np.zeros(Nf,float)
                arr_rezBzz=arr_rezBz.copy()
                arr_rezMx=  np.zeros((Ngroup,Nf),float)
                arr_rezMn=  np.zeros((Ngroup,Nf),float)
                Arr_AAA=np.zeros((NIter*Nproc,Nf),float) 
                all_rezAz=np.zeros((NIter,Nf),float)
                arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]  
                all_RezN=np.zeros((Ngroup,NIter,Nf),float)+1e32
                all_RezM=np.zeros((Ngroup,NIter,Nf),float)-1e32
                dd1a=np.zeros((Ngroup,NIter,Nf),float)-1e32
                dd2a=np.zeros((Ngroup,NIter,Nf),float)+1e32
                all_RezNM=np.zeros((Ngroup,NIter,Nf),float)
                all_RezMM=np.zeros((Ngroup,NIter,Nf),float)
                hhh_=hhh_+1
            else:
                hhh_=hhh_+1
                ZZ=1
        
        if ZZ==0:                  
            try:
                [hhha,Arr_AAA]=(hkl.load(wrkdir + aname+".rlf1"))       
            except:            
                hhha=hhh-1
                           
            #hhha=NIter 
            if hhh>=hhha: 
                if Lo:
                    arr_A=np.log(arr_z) 
                else:
                    arr_A=arr_z.copy()
                    
                Asr=np.mean(arr_A[0:Nf-NNew])
                arr_A=arr_A-Asr
                Klg=np.power(10,np.floor(np.log10(np.max(abs(arr_A)))))
                arr_A=arr_A/Klg
        
                program =wrkdir + "RALF1FiltrX_lg.py"
                NChan=1
                for iProc in range(Nproc):
                    argss[iProc]=["%s"%iProc, "%s"%NChan, "%s"%NNew, "%s"%NIt]#"%s"%(iProc+1)]
                    for i in range(Nf):
                        argss[iProc].append(str("%1.6f"%(arr_A[i])))
                
                arezAMx=[] 
                # for iProc in range(Nproc):
                #     arezAMx.append(RALf1FiltrQ(argss[iProc]))
                
                pool = mp.Pool(processes=Nproc,)
                arezAMx.append(pool.map(RALf1FiltrQ, argss))
                arezAMx= np.asarray(arezAMx,float)[0,:,:]
                del(pool)

                # executor=concurrent.futures.ThreadPoolExecutor(max_workers=Nproc)
                # future_to = {executor.submit(RALf1FiltrQ, argss[iProc]) for iProc in range(Nproc)}
                # ii=0
                # for future in concurrent.futures.as_completed(future_to): 
                #     print("Result %d\n"%(ii+1))
                #     ii=ii+1
                #     arezAMx.append(np.asarray(future.result(),float))
                #     if len(arezAMx)>=Nproc:
                #         break
                # del(future)                        
                # del(executor)
                
                arezAMx= np.asarray(arezAMx,float)*Klg+Asr
                 
                for iGr in range(Ngroup):
                    Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+hhh*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)]=(
                        arezAMx[int(iGr*(Nproc/Ngroup)):int((iGr+1)*(Nproc/Ngroup))]).copy()
                                             
                hkl.dump([hhh+1,Arr_AAA], wrkdir + aname+".rlf1")  
                [hhha,Arr_AAA]=(hkl.load(wrkdir + aname+".rlf1"))
            
            WrtTodr=0
            if hhh>=hhha-1:   
                WrtTodr=1
                aDur=4
                                
            aNN=2
            aMM=3
            
            ar0x=ar0.copy()
            if not sum(abs(arr_rezBzz))==0:
                ar0x=arr_rezBzz.copy()
            
            if Lo:
                dd=np.log(arr_z) 
                dd_=np.log(ar0x)                   
            else:
                dd=arr_z.copy()  
                dd_=ar0x.copy()
                
            astart0=np.Inf
            astart0=dd[0]
            dd_[1:]=np.diff(dd_)
            dd_[0]=0
            if Lo:
                ar0_=np.exp(dd_)
            else:
                ar0_=dd_.copy()
            MMM_=0
            while MMM_<2*Nproc:              
                arr_RezM=  np.zeros((Ngroup,Nf),float)  
                arr_RezN=  np.zeros((Ngroup,Nf),float)  
                MMM=0
                for iGr in range(Ngroup):                
                    ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
                    xxxx=ZDat.copy()
                    for i in range(aNN-1):
                        xxxx=np.concatenate((xxxx, ZDat))
                    ZDat=xxxx.copy()
                    hhhx=0
                    anI=len(ZDat)
            
                    for i in range(anI):
                        if not astart0==np.Inf:
                            if Lo:
                                ZDat[i][0:Nf-NNew]=np.log(arr_z[0:Nf-NNew]) 
                                if hhh==0:
                                    ZDat[i][1:]=np.diff(ZDat[i])
                                else:
                                    ZDat[i][1:]=np.diff(ZDat[i]+KPP*np.log(arr_rezBzz))
                            else:
                                ZDat[i][0:Nf-NNew]=arr_z[0:Nf-NNew].copy() 
                                if hhh==0:
                                    ZDat[i][1:]=np.diff(ZDat[i])
                                else:
                                    ZDat[i][1:]=np.diff(ZDat[i]+KPP*arr_rezBzz)
                            ZDat[i][0]=0     
                            #ZDat[i][Nf-NNew:]=0
                        else:
                            if Lo:
                                ZDat[i][0:Nf-NNew]=np.log(arr_z[0:Nf-NNew]) 
                                if not hhh==0:
                                    ZDat[i]=(ZDat[i]+KPP*np.log(arr_rezBzz))
                            else:
                                ZDat[i][0:Nf-NNew]=arr_z[0:Nf-NNew].copy() 
                                if not hhh==0:
                                    ZDat[i][1:]=(ZDat[i]+KPP*arr_rezBzz)
            
                    if anI<aNN: 
                        all_RezM[iGr][hhh]=np.amax(ZDat,axis=0)
                        all_RezN[iGr][hhh]=np.amin(ZDat,axis=0)
                    else:
                        aMx_=0
                        aMn_=0
                        mdd4=ZDat*0
                        aa=RandomQ(Nf)                        
                        ss0=np.concatenate((aa, aa, aa))
                        hhhx=0
                        while hhhx<int(NIter/1):
                            #if dNIt*int(hhhx/dNIt)==hhhx:
                            mdd4=mdd4*0
                            dd=ZDat.copy()                        
                            aa=RandomQ(Nf)                        
                            ss4=np.concatenate((aa, aa, aa))
                            liix=np.zeros((anI,Nf),int)
                            mdd4_=np.zeros(Nf,float)
                            mdd4_[0:Nf-NNew]=1
                            for i in range(anI):  
                                liix[i]=ss4[ss0[i+hhhx:i+Nf+hhhx]].copy()
                                dd[i]=(dd[i])[liix[i]].copy() 
                                mdd4[i]=mdd4_[liix[i]].copy()
                                
                            astart=np.Inf
                            dd=dd.reshape((anI*Nf))  
                            astart=dd[0]
                            dd[1:]=np.diff(dd)
                            dd[0]=0                        
                            dd=dd.reshape((anI,Nf))                                 
                            D=np.std(dd)
                            
                            aa=RandomQ(Nf)                        
                            ss4_=np.concatenate((aa, aa, aa))                                      
                            DD_=[]
                            for hhhc in range(anI):
                                DD_.append(ss4_[hhhc+Nf:hhhc:-1].copy())
                            DD_=np.asarray(DD_,float)                              
                            DD_=(DD_/np.std(DD_))*D
                            DD_=(DD_-np.mean(DD_))
                                                    
                            dd_AA=dd.copy()
                            dd_BB=dd.copy()
                            dd_CC=dd.copy()
                            P=np.zeros(3,float)
                            PP=1
                            
                            seq0=(dd.reshape(len(dd)*len(dd[0])))[1:]*np.ceil(0.5*(1/(mdd4.reshape(len(dd)*len(dd[0]))==1)[0:len(dd)*len(dd[0])-1]+1/(mdd4.reshape(len(dd)*len(dd[0]))==1)[1:]))
                            seq0=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seq0)),float) 
                            seq0=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seq0)),float)    
                            
                            for ii in range(aNN):   
                                for jj in range(aMM):
                                    dd1=dd[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)].copy()
                                    mdd4_=mdd4[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)].copy()
                                    
                                    seqA=(dd1.reshape(len(dd1)*len(dd1[0])))[1:]*np.ceil(0.5*(1/(mdd4_.reshape(len(dd1)*len(dd1[0]))==1)[0:len(dd1)*len(dd1[0])-1]+1/(mdd4_.reshape(len(dd1)*len(dd1[0]))==1)[1:]))
                                    seqA_=  mdd4_.reshape(len(dd1)*len(dd1[0]))*0
                                    seqA_[0]=1
                                    seqA_[1:]=(abs(seqA)==np.Inf)+np.isnan(seqA)
                                    seqA_=seqA_.reshape((len(dd1),len(dd1[0])))
                                    seqA=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqA)),float) 
                                    seqA=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqA)),float)    
                                    
                                    DD__A=DD_[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)].copy()
                                    DD__B=-DD__A[:,::-1].copy()
                                    if len(dd1)>1 and len(dd1[0])>=len(dd1):
                                        eeA= (XFilter.RALF1FilterX( dd1*(1-seqA_)+seqA_*( dd1-(DD__A)),len(dd1),len(dd1[0]),1,0))
                                        eeB=-(XFilter.RALF1FilterX(-dd1*(1-seqA_)+seqA_*(-dd1-(DD__B)),len(dd1),len(dd1[0]),1,0))
                                        dd_AA[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=eeB.copy()#*(eeB>0)*((eeA+eeB)>0)
                                        dd_BB[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=eeA.copy()#*(eeA<0)*((eeA+eeB)<0)
                          
                                    dd2=0.25*(dd_AA+dd_BB)[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]
                                    seqB=(dd2.reshape(len(dd2)*len(dd2[0])))[1:]*np.ceil(0.5*(1/(mdd4_.reshape(len(dd2)*len(dd2[0]))==1)[0:len(dd2)*len(dd2[0])-1]+1/(mdd4_.reshape(len(dd2)*len(dd2[0]))==1)[1:]))
                                    seqB=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqB)),float) 
                                    seqB=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqB)),float)    
            
                                    P[0:2]=np.polyfit(seqA,seqB,1)
                                    try:
                                        if P[0]>0 and 100*scp.pearsonr(seqA,seqB)[0]>50:
                                            dd_CC[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=(dd2.copy()-P[1])/P[0] 
                                        else:
                                            PP=0
                                    except:
                                        PP=0
                                    # dd_CC[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=(dd2.copy()) 
                            
                            seq0_=(dd_CC.reshape(len(dd_CC)*len(dd_CC[0])))[1:]*np.ceil(0.5*(1/(mdd4.reshape(len(dd_CC)*len(dd_CC[0]))==1)[0:len(dd_CC)*len(dd_CC[0])-1]+1/(mdd4.reshape(len(dd_CC)*len(dd_CC[0]))==1)[1:]))
                            seq0_=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seq0_)),float) 
                            seq0_=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seq0_)),float)    
                            
                            P[0:2]=np.polyfit(seq0,seq0_,1)
                            if P[0]>0 and 100*scp.pearsonr(seq0,seq0_)[0]>50:
                                dd_CC=(dd_CC-P[1])/P[0] 
                            else:
                                PP=0
                            
                            dd_AA=dd_CC*PP*(dd_CC>0)
                            dd_BB=dd_CC*PP*(dd_CC<0)                                                                                     
                            if not astart==np.Inf: 
                                dd_AA=dd_AA.reshape((anI*Nf))
                                dd_AA=astart+np.cumsum(dd_AA)
                                dd_AA=dd_AA.reshape((anI,Nf))
                                dd_BB=dd_BB.reshape((anI*Nf))
                                dd_BB=astart+np.cumsum(dd_BB)
                                dd_BB=dd_BB.reshape((anI,Nf))
                                
                            dd_AA=(dd_AA+dd_BB)
                            dd_BB=dd_AA.copy()  
                            aMx=np.zeros(Nf,float)-1e32
                            aMn=np.zeros(Nf,float)+1e32
                            for i in range(anI):
                                aMx[liix[i]]=np.maximum(aMx[liix[i]],dd_AA[i])
                                aMn[liix[i]]=np.minimum(aMn[liix[i]],dd_BB[i])
                            
                            if dNIt*int(hhhx/dNIt)==hhhx:
                                aMx_=aMx.copy()
                                aMn_=aMn.copy()
                                aMx0=aMx_.copy()
                                aMn0=aMn_.copy()
                            
                            # aMx_=np.maximum(aMx_,aMx)
                            # aMn_=np.minimum(aMn_,aMn)
                            aMx_=(aMx_*(hhhx-dNIt*int(hhhx/dNIt))+np.maximum(aMx_,aMx))/(hhhx-dNIt*int(hhhx/dNIt)+1)
                            aMn_=(aMn_*(hhhx-dNIt*int(hhhx/dNIt))+np.minimum(aMn_,aMn))/(hhhx-dNIt*int(hhhx/dNIt)+1)
                            
                            if Lo:
                                arr_RezM[iGr][Nf-NNew:]=(filterFourierQ(aMx_,np.log(ar0_),NNew,1,1)[Nf-NNew:]+
                                    filterFourierQ(aMn_,np.log(ar0_),NNew,1,1)[Nf-NNew:])/2
                            else:
                                arr_RezM[iGr][Nf-NNew:]=(filterFourierQ(aMx_,(ar0_),NNew,1,1)[Nf-NNew:]+
                                    filterFourierQ(aMn_,(ar0_),NNew,1,1)[Nf-NNew:])/2
                            y=arr_RezM[iGr][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                            if Lo:
                                x=np.log(ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                            else:
                                x=ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                            P[0:2]=np.polyfit(x,y,1)
                            if not (P[0]>0 and 100*scp.pearsonr(x,y)[0]>10):
                                PP=0
                            if not PP==0:
                                # P[0]=np.std(y)/np.std(x)
                                # P[1]=np.mean(y)-P[0]*np.mean(x)
                                arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-P[1])/P[0] 
                                if Lo:
                                    arr_RezM[iGr][:Nf-NNew]=np.log(ar0_[:Nf-NNew])
                                else:
                                    arr_RezM[iGr][:Nf-NNew]=ar0_[:Nf-NNew].copy()
                                
                                aMx0=aMx_.copy()
                                aMn0=aMn_.copy()
                                arr_RezN[iGr]=arr_RezM[iGr].copy()
                                                               
                                if dNIt*int(hhhx/dNIt)==hhhx:
                                    all_RezM[iGr][hhh]=arr_RezM[iGr].copy()
                                    all_RezN[iGr][hhh]=arr_RezN[iGr].copy()
                                allaMx=np.maximum(all_RezM[iGr][hhh],arr_RezM[iGr])
                                allaMn=np.minimum(all_RezN[iGr][hhh],arr_RezN[iGr])
                                y=0.5*(allaMx+allaMn)[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                P[0:2]=np.polyfit(x,y,1)
                                if P[0]>0 and not PP==0:
                                    all_RezM[iGr][hhh]=allaMx.copy()
                                    all_RezN[iGr][hhh]=allaMn.copy()                                    
                                    if hhhx==0:
                                        dd1a[iGr,hhh]=(all_RezM[iGr][hhh]-P[1])/P[0] 
                                        dd2a[iGr,hhh]=(all_RezN[iGr][hhh]-P[1])/P[0] 
                                    else:
                                        dd1a[iGr,hhh]=(dd1a[iGr,hhh]*hhhx+(all_RezM[iGr][hhh]-P[1])/P[0] )/(hhhx+1)
                                        dd2a[iGr,hhh]=(dd2a[iGr,hhh]*hhhx+(all_RezN[iGr][hhh]-P[1])/P[0] )/(hhhx+1)                                
                                    hhhx=hhhx+1
                            if PP==0:
                                aMx_=aMx0.copy()
                                aMn_=aMn0.copy()
                                aa=RandomQ(Nf)                        
                                ss0=np.concatenate((aa, aa, aa))
                                        
                    dd1=np.amax(dd1a[iGr,max(0,(hhh+1)-int(dNIt/2+1)):hhh+1],axis=0)
                    dd2=np.amin(dd2a[iGr,max(0,(hhh+1)-int(dNIt/2+1)):hhh+1],axis=0)
                    
                    asr1=abs(dd1)>abs(dd2)
                    asr2=abs(dd1)<abs(dd2)
                    #arr_RezM[iGr]=(dd1+dd2)/2#
                    arr_RezM[iGr]=dd1*asr1+dd2*asr2+(dd1+dd2)*(asr1==asr2)/2
                    if Lo:
                        arr_RezM[iGr][Nf-NNew:]=(filterFourierQ((arr_RezM[iGr]),np.log(ar0_),NNew,1,1))[Nf-NNew:]
                    else: 
                        arr_RezM[iGr][Nf-NNew:]=(filterFourierQ((arr_RezM[iGr]),(ar0_),NNew,1,1))[Nf-NNew:]
                    all_RezNM[iGr][hhh]=arr_RezM[iGr].copy()
                    if Lo:
                        x=np.log(ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                    else:
                        x=ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                    
                    y=all_RezNM[iGr][hhh][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                    P[0:2]=np.polyfit(x,y,1)
                    # P[0]=np.std(y)/np.std(x)
                    # P[1]=np.mean(y)-P[0]*np.mean(x)
                    all_RezNM[iGr][hhh][Nf-NNew:]=(all_RezNM[iGr][hhh][Nf-NNew:]-P[1])/P[0] 
                        
                    if not astart0==np.Inf:
                        all_RezMM[iGr][hhh]=np.cumsum(all_RezNM[iGr][hhh])
                    else:
                        all_RezMM[iGr][hhh]=all_RezNM[iGr][hhh].copy()
                    
                    if Lo:
                        all_RezMM[iGr][hhh][Nf-NNew:]=(filterFourierQ((all_RezMM[iGr][hhh]),np.log(ar0x),NNew,1,1))[Nf-NNew:]
                    else: 
                        all_RezMM[iGr][hhh][Nf-NNew:]=(filterFourierQ((all_RezMM[iGr][hhh]),(ar0x),NNew,1,1))[Nf-NNew:]
                    
                    if Lo:
                        x=np.log(ar0[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                    else:
                        x=ar0[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                
                    y=all_RezMM[iGr][hhh][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()                                       
                    
                    P[0:2]=np.polyfit(x,y,1)
                    P[0]=np.std(y)/np.std(x)
                    P[1]=np.mean(y)-P[0]*np.mean(x)
                    
                    if 100*scp.pearsonr(x,all_RezMM[iGr][hhh][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])[0]<20:
                        MMM=MMM+1
                    all_RezMM[iGr][hhh][Nf-NNew:]=(all_RezMM[iGr][hhh][Nf-NNew:]-P[1])/P[0]
                    if Lo:
                        all_RezMM[iGr][hhh][:Nf-NNew]=np.log(ar0[:Nf-NNew])
                    else:
                        all_RezMM[iGr][hhh][:Nf-NNew]=ar0[:Nf-NNew].copy()
                    
                    arr_RezM[iGr]=np.mean(all_RezMM[iGr][0:hhh+1],axis=0)
                
                MMM=int(2*MMM/Ngroup)
                arr_rezBz=np.mean(arr_RezM, axis=0) 
                if Lo:
                    arr_rezBz=np.exp(arr_rezBz) 
                    for iGr in range(Ngroup): 
                        arr_RezM[iGr]=np.exp((arr_RezM[iGr]))  
                        
                mm1=ar0[Nf-NNew:].copy()                            
                mm2=arr_rezBz[Nf-NNew:len(ar0)].copy()   
                try: 
                    if 100*scp.pearsonr(mm1,mm2)[0]>10 and MMM==0:
                        break
                    else:
                        MMM_=MMM_+1
                except:
                    break

            if MMM_<2*Nproc and np.std(mm1)>0 and np.std(mm2)>0:
                arr_rezBzz=arr_rezBz.copy()
                Koef_.append(100*scp.pearsonr(mm1,mm2)[0])                               
                fig = plt.figure()
                axes = fig.add_axes([0.1, 0.1, 1.2, 1.2])
                axes_ = fig.add_axes([0, 0, 0.3, 0.3]) 
                axes__ = fig.add_axes([0.4, 0, 0.3, 0.3])        
                
                if Lo:
                    axes.semilogy(ar0, 'ro-', alpha=0.1)
                    axes.semilogy(arrr, 'rx-')
                    for iGr in range(Ngroup):
                        axes.semilogy(arr_RezM[iGr],linewidth=3.,alpha=0.2)
                    axes.semilogy(arr_rezBz,'yx-',linewidth=4.,alpha=0.5)
                    axes.grid(True, which="both", ls="-")
                else:
                    axes.plot(ar0, 'ro-', alpha=0.1)
                    axes.plot(arrr, 'rx-')
                    for iGr in range(Ngroup):
                        axes.plot(arr_RezM[iGr],linewidth=3.,alpha=0.2)
                    axes.plot(arr_rezBz,'yx-',linewidth=4.,alpha=0.5)
                    axes.grid(True, which="both", ls="-")
                
                axes.text(-0.1, 4, '%s  '%(hhh+1),
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=axes_.transAxes,color='black', fontsize=24) 
                axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s, start = %s, step = %s * %s'%(aname,adat0,interv,aDecm),
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axes_.transAxes,color='blue', fontsize=14)    
                if Lo:
                    gint=np.polyfit(np.log(mm1),np.log(mm2),1)
                else:
                    gint=np.polyfit(mm1,mm2,1)
                
                if Lo:                
                    axes_.loglog(mm1,np.exp(gint[1]+gint[0]*np.log(mm1)),'y.',linewidth=2.)                    
                    axes_.loglog(mm1,mm2, 'ok', markersize=3, alpha=0.1) 
                else:
                    axes_.plot(mm1,gint[1]+gint[0]*mm1,'y.',linewidth=2.)                    
                    axes_.plot(mm1,mm2, 'ok', markersize=3, alpha=0.1) 
                
                axes_.text(0.2, 0.6, '%d'%int(np.floor(np.asarray(Koef_,float)[::-1][0])),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=axes_.transAxes,color='green', fontsize=30)
                
                axes__.text(1.8, 0.6, 'Dunning-Kruger\n effect',
                        verticalalignment='bottom', horizontalalignment='center',
                    transform=axes_.transAxes,color='green', fontsize=14)  
                axes__.plot(np.asarray(range(hhh+1),float)+1,Koef_,'y',linewidth=2.)
                
                frame=fig2img(fig) 
                cimg = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)        
                gray_sz1=min(gray_sz1,len(cimg[0]))
                gray_sz2=min(gray_sz2,len(cimg))
                ImApp.append(frame)
                if WrtTodr>0 or 10*int((hhh+1)/10)==(hhh+1) or hhha==(hhh+1):
                    out = cv.VideoWriter(wrkdir + aname+'.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
                    for icl in range(len(ImApp)):
                        cimgx=(cv.cvtColor(np.array(ImApp[icl]), cv.COLOR_RGB2BGR)) 
                        out.write(cimgx[0:gray_sz2,0:gray_sz1,:]) 
                    out.release()
                    del(out)
                plt.show()
                hhh=hhh+1
                #arr_z=arr_rezBz.copy()
            else:
                try:
                    dill.load_session(wrkdir + aname+".ralf")
                except:
                    hh0=hh0                  
            hh0=hh0+1
            if WrtTodr>0:
                try:
                    dill.dump_session(wrkdir + aname+".ralf")  
                except:
                    hh0=hh0
            if hh0==2*NIter:
                hhh=NIter 
            print (hhh+10000*hh0)
            
                
#        df = pd.DataFrame(arr_rezBz)
#        df.to_excel (wrkdir +r'export_traces.xlsx', index = None, header=False) 
    
    mm1=arrr[len(arrr)-int(len(arrr)/2):].copy()                            
    mm2=arr_rezBz[len(arrr)-int(len(arrr)/2):len(arrr)].copy()  
    if np.std(mm1)>0 and np.std(mm2)>0:
        Koef=100*scp.pearsonr(mm1,mm2)[0] 
        
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 1.2, 1.2])
    axes_ = fig.add_axes([0, 0, 0.3, 0.3])
    axes__ = fig.add_axes([0.4, 0, 0.3, 0.3]) 
    
    if Lo:
        axes.semilogy(arrr, 'ro-')
        axes.semilogy(arr_rezBz, 'cx-', alpha=0.5) #predicted data
        axes.grid(True, which="both", ls="-")
    else:
        axes.plot(arrr, 'ro-')
        axes.plot(arr_rezBz, 'cx-', alpha=0.5) #predicted data
        axes.grid(True, which="both", ls="-")
    
    axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s, start = %s, step = %s * %s'%(aname,adat0,interv,aDecm),
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes_.transAxes,color='blue', fontsize=14)     
    if Lo:
        gint=np.polyfit(np.log(mm1),np.log(mm2),1)
    else:
        gint=np.polyfit(mm1,mm2,1)
    
    if Lo:
        axes_.loglog(mm1,np.exp(gint[0]+gint[1]*np.log(mm1)),'y',linewidth=2.) 
        axes_.loglog(mm1,mm2, 'ok', markersize=3, alpha=0.1)    
    else:
        axes_.plot(mm1,gint[0]+gint[1]*mm1,'y',linewidth=2.) 
        axes_.plot(mm1,mm2, 'ok', markersize=3, alpha=0.1)    
    
    axes_.text(0.2, 0.6, '%d'%int(np.floor(np.asarray(Koef_,float)[::-1][0])),
        verticalalignment='bottom', horizontalalignment='right',
        transform=axes_.transAxes,color='green', fontsize=30)    
    
    axes__.text(1.8, 0.6, 'Dunning-Kruger\n effect',
        verticalalignment='bottom', horizontalalignment='center',
                transform=axes_.transAxes,color='green', fontsize=14)  
    axes__.plot(Koef_,'y',linewidth=2.)
    
    frame=fig2img(fig) 
    cimg = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)        
    gray_sz1=min(gray_sz1,len(cimg[0]))
    gray_sz2=min(gray_sz2,len(cimg))
    for icl in range(10):
        ImApp.append(frame)
    out = cv.VideoWriter(wrkdir + aname+'.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
    for icl in range(len(ImApp)):
        cimgx=(cv.cvtColor(np.array(ImApp[icl]), cv.COLOR_RGB2BGR)) 
        out.write(cimgx[0:gray_sz2,0:gray_sz1,:])       
    out.release()
    plt.show()
    del(out)

