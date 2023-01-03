#import concurrent.futures
import pylab as plt
import numpy as np
import urllib.request, json 
import pandas as pd
from PIL import Image  
import cv2 as cv
import time as tm 
import hickle as hkl
import os
import multiprocessing as mp

from scipy import stats as scp
import dateutil.parser
from operator import itemgetter
import dill 
#from scipy.signal import savgol_filter
#import wmi as wm
import csv
from RALf1FiltrVID import filterFourierQ
from RALf1FiltrVID import RALf1FiltrQ
from RALf1FiltrVID import RandomQ
import RALF1FilterX as XFilter  

MxTime=1*60*60 # 2 haurs
#https://query1.finance.yahoo.com/v7/finance/download/LRC-USD?period1=1635554377&period2=1667097577&interval=1d&events=history&includeAdjustedClose=true
wrkdir = r"/home/vacho/Документи/Work/W14_7/WX4/"
api_key = 'ONKTYPV6TAMZK464' 
 
interv="15min"
interv="Daily"

#INTRADAY
#d_intervals = {"1min","5min","15min","30min","60min"}

Lengt0=700
Ngroup=3
Nproc=2*Ngroup#*(os.cpu_count())
Lo=1  
lSrez=0.99
aTmStop=1
NIt=3
NIter=100
DT=0.3
dNIt=4
aDecm=3
KPP=0

aKEY=0
def decimat(adat_):
    if Lo:
        if sum(adat_<=0)==0:
            adat_=np.log(adat_)
        else:
            return 0
                
    adatx=0
    k=0
    adat__=np.zeros(int(len(adat_)/aDecm),float)
    for i in range(int(len(adat_)/aDecm)):
        adat__[k]=np.mean(adat_[i*aDecm:i*aDecm+aDecm])
        k=k+1
    if Lo:
        return np.exp(adat__[1:len(adat__)])
    else:
        return (adat__[1:len(adat__)])

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
        with open(wrkdir+ticker1+ '.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            dat=[]
            i=0
            for row in spamreader:
                if i>0:
                    dat.append(row[2])
                i=i+1
        dat=dat[len(dat)-aLengt+2*aDecm:]
        dat=dat[:len(dat)]
        try:
            dat=np.asarray(dat,float)
            arrr=np.asarray(dat,float)
        except:
            dat=dat
            arrr=np.zeros(2,int)
        
    return arrr,adat_

try:
    nnams_=hkl.load(wrkdir +"name.rlf1")
    ii=len(nnams_)
except:
    WhO=[
"BTC-USD", 
"ETH-USD", 
"ADA-USD", 
"SOL-USD", 
"DOGE-USD", 
"UNI1-USD", 
"LINK-USD", 
"BCH-USD", 
"LTC-USD", 
"SHIB-USD", 
"MATIC-USD", 
"XLM-USD", 
"ETC-USD", 
"ATOM-USD", 
"EOS-USD", 
"AVE-USD", 
"GRT1-USD", 
"XTZ-USD", 
"MKR-USD", 
"COMP1-USD", 
"MINA-USD", 
"SUSHI-USD", 
"SNX-USD", 
"OMG-USD", 
"BNT-USD", 
"CRV-USD", 
"ZRX-USD", 
"UMA-USD", 
"CELO-USD", 
"ANKR-USD", 
"KNC-USD", 
"LRC-USD", 
"SKL-USD", 
"STORJ-USD", 
"NU-USD", 
"NMR-USD", 
"BAL-USD", 
"BAND-USD", 
"AXS-USD", 
"ICP-USD", 
"IOTX-USD", 
"ORN-USD",
"DOT-USD"
    ]
    
    arrrxxR=[]
    nams=[]
    lenar=[]

    for i in range(len(WhO)):
        i0=0
        while i0<6 and not i0<0:
            try:
                arrrxx1,adat1_=loaddata(Lengt0,WhO[i],aKEY)
                arrrxxR.append(arrrxx1)
                lenar.append(len(arrrxx1))
                nams.append(WhO[i])
                i0=-1
            except:
                i0=i0+1
                tm.sleep(1)
               
    llar=int(0.99*np.median(np.asarray(lenar,int)))
    nnams_=[]
    aaer=[]

    for i in range(len(nams)):    
        if len(arrrxxR[i])>=llar:
            aer=decimat(arrrxxR[i])
            try:
                if len(aer)>0:
                    nnams_.append(nams[i])
                    aaer.append(aer[len(aer)-int(llar/aDecm)+1:].copy())
            except:
                aer=aer
            
    arrrxxR_=np.asarray(aaer,float)
               
    aKEY=1
    for uuii in range(len(nnams_)):
        aname=nnams_[uuii]
        ticker=aname+"YLLL"
        ticker1=aname
        ticker2=aname
        try:                
            hkl.dump(arrrxxR_[uuii],wrkdir + aname+"dat.rlf1")
        except:
            os.mkdir(wrkdir)
            hkl.dump(arrrxxR_[uuii],wrkdir + aname+"dat.rlf1")
    hkl.dump(nnams_,wrkdir + "name.rlf1")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
nnams__=nnams_.copy()
Nii=len(nnams__)
if __name__ == '__main__': 
    # w = wm.WMI(namespace="OpenHardwareMonitor")
    # temperature_infos = w.Sensor()
    # if temperature_infos==[]:
    #     os.startfile(r".\\OpenHardwareMonitor\OpenHardwareMonitor.exe")
    # del(w)
    # del(temperature_infos)
    # del(wm)
    for uuii in range(Nii):
        nnams_=hkl.load(wrkdir + "name.rlf1")
        Nii=len(nnams_)
        aname=nnams_[uuii]
        ticker=aname+"YLLL"
        ticker1=aname
        ticker2=aname
        try:
            dill.load_session(wrkdir + aname+".ralf")
            nnams_=hkl.load(wrkdir + "name.rlf1")
            Nii=len(nnams_)
        except:
            ImApp=[]
            try:
                arrrxx=hkl.load(wrkdir + aname+"dat.rlf1")
            except:
    
                #ticker ="EOSOMG" # "BTCUSD"#"GLD"#"DJI","LOIL.L"#""BZ=F" "LNGA.MI" #"BTC-USD"#"USDUAH"#"LTC-USD"#"USDUAH"#
                if not ticker1==ticker2:
                    arrrxx1,adat1_=loaddata(Lengt0,ticker1,aKEY)
                    arrrxx1=np.asarray(arrrxx1,float)
                    arrrxx2,adat2_=loaddata(Lengt0,ticker2,aKEY)
                    arrrxx2=np.asarray(arrrxx2,float)
                    lnm=min(len(arrrxx1),len(arrrxx2))
                    arrrxx=arrrxx1[len(arrrxx1)-lnm:]/arrrxx2[len(arrrxx2)-lnm:]
                else:
                    ticker=ticker1
                    arrrxx,adat_=loaddata(Lengt0,ticker,aKEY)
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
    
            arrr=np.asarray(arrrxx).copy()  
    
            arrr=np.asarray(arrr,float)    
            Lengt=len(arrr)
            Nf=Lengt
            
            nn=int(Nf*DT)             
            NNew=int(Nf*0.5)  
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
            all_RezN=np.zeros((Ngroup,NIter,Nf),float)+np.Inf
            all_RezM=np.zeros((Ngroup,NIter,Nf),float)-np.Inf
            dd1a=np.zeros((Ngroup,NIter,Nf),float)-np.Inf
            dd2a=np.zeros((Ngroup,NIter,Nf),float)+np.Inf
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
                nnams_=hkl.load(wrkdir + "name.rlf1")
                Nii=len(nnams_)
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
                
        #key=13
        while hhh_<aTmStop and not key == 13: 
            Aprocess=[]
            if hhh==int(NIter/33):
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
                if hh0>=hhha: 
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
                        argss[iProc].append("%d"%Nproc)                    
                    
                    try:
                        arezAMx_=hkl.load("ralfrez.rlf2")
                        if len(arezAMx_)==Nproc:
                            hkl.dump([],"ralfrez.rlf2")
                    except:
                        hkl.dump([],"ralfrez.rlf2")
                    
                    arezAMx_=[] 
                    # for iProc in range(Nproc):
                    #     aaa=RALf1FiltrQ(argss[iProc])
                    #     arezAMx_.append(aaa)
                    
                    pool = mp.Pool(processes=Nproc)
                    pool.map(RALf1FiltrQ, argss)
                    #arezAMx= np.asarray(arezAMx,float)[0,:,:]
                    del(pool)

                    # with concurrent.futures.ThreadPoolExecutor(max_workers=Nproc) as executor:
                    #     future_to = {executor.submit(RALf1FiltrQ, argss[iProc]): iProc for iProc in range(Nproc)}
                    #     ii=0
                    #     for future in concurrent.futures.as_completed(future_to):                    
                    #         arezAMx_.append(np.asarray(future.result(),float))
                    #         ii=ii+1
                    #         if ii==Nproc:
                    #             executor.shutdown()
                    # del(future)                        
                    # del(executor)
                    
                    arezAMx_=hkl.load("ralfrez.rlf2")
                    if len(arezAMx_)>0:
                        hkl.dump(arezAMx_,"ralfrez_.rlf2")
                    arezAMx= np.asarray(arezAMx_,float)
                    hkl.dump([],"ralfrez.rlf2")
                    
                    
                    arezAMx= np.asarray(arezAMx,float)*Klg+Asr
                     
                    for iGr in range(Ngroup):
                        Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+hhh*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)]=(
                            arezAMx[int(iGr*(Nproc/Ngroup)):int((iGr+1)*(Nproc/Ngroup))]).copy()
                                                 
                    hkl.dump([hh0+1,Arr_AAA], wrkdir + aname+".rlf1")  
                    [hhha,Arr_AAA]=(hkl.load(wrkdir + aname+".rlf1"))
                
                WrtTodr=1
                if hhh>=hhha-1:   
                    WrtTodr=1
                    aDur=4
                                    
                aNN=2
                aMM=2
                for iGr in range(Ngroup):                
                    ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hhh+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
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
                    ar0x_=2*np.median(abs(ZDat-np.log(ar0x)),axis=0)
                else:
                    ar0x=np.median(ZDat,axis=0)
                    ar0x_=2*(np.median(abs((ZDat)-(ar0x)),axis=0))

                for i in range(anI):    
                    if Lo:                                
                        ZDat[i]=(ZDat[i]*(abs(ZDat[i]-np.log(ar0x))<=ar0x_))+np.log(ar0x)*(abs(ZDat[i]-np.log(ar0x))>ar0x_)
                    else:
                        ZDat[i]=(ZDat[i]*(abs((ZDat[i])-(ar0x))<=ar0x_))+ar0x*(abs((ZDat[i])-(ar0x))>ar0x_)
                if Lo:
                    ar0x=np.exp(np.median(ZDat,axis=0))  
                    ar0x_=2*np.median(abs(ZDat-np.log(ar0x)),axis=0)
                    ZDat=np.exp(ZDat)
                else:
                    ar0x=np.median(ZDat,axis=0)
                    ar0x_=2*(np.median(abs((ZDat)-(ar0x)),axis=0))        
                
                #ar0x[0:len(ar0)]=ar0[0:len(ar0)].copy()
                # ar0x=ar0.copy()
                # if not sum(abs(arr_rezBzz))==0:
                #     ar0x=arr_rezBzz.copy()
               
                if Lo:
                    dd_=np.log(ar0x)                   
                else:
                    dd_=ar0x.copy()
                    
                astart0=np.Inf
                # astart0=dd_[0]
                # dd_[1:]=np.diff(dd_)
                # dd_[0]=0
                if Lo:
                    ar0_=np.exp(dd_)
                else:
                    ar0_=dd_.copy()
                MMM_=0
                tm0=tm.time()
                tm1=0
                aMx0=dd_*0-np.Inf
                aMn0=dd_*0+np.Inf
                while MMM_<2*Nproc and (tm1-tm0)<MxTime:              
                    arr_RezM=  np.zeros((Ngroup,Nf),float)  
                    arr_RezN=  np.zeros((Ngroup,Nf),float)  
                    MMM=0
                    for iGr in range(Ngroup):
                        if Lo:
                            ZDat=np.exp(Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+hhh*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)])
                        else:
                            ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+hhh*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hhh+1)*int(Nproc/Ngroup)].copy()
                            
                    # for iGr in range(Ngroup):                
                    #     if Lo:
                    #         ZDat=np.exp(Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hh0+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hh0+1)*int(Nproc/Ngroup)])
                    #     else:
                    #         ZDat=Arr_AAA[iGr*NIter*int(Nproc/Ngroup)+max(0,(hh0+1)-dNIt)*int(Nproc/Ngroup):iGr*NIter*int(Nproc/Ngroup)+(hh0+1)*int(Nproc/Ngroup)].copy()
                        xxxx=ZDat.copy()
                        for i in range(aNN-1):
                            xxxx=np.concatenate((xxxx, ZDat))
                        ZDat=xxxx.copy()
                        hhhx=0
                        anI=len(ZDat)
                        # for i in range(anI):  
                        #     ZDat[i][:len(ar0)]=ar0.copy()
                        for i in range(anI):
                            if Lo:                                
                                ZDat[i]=np.exp(np.log(ZDat[i])*(abs(np.log(ZDat[i])-np.log(ar0_))<=ar0x_)+np.log(ar0_)*(abs(np.log(ZDat[i])-np.log(ar0_))>ar0x_))
                            else:
                                ZDat[i]=(ZDat[i]*(abs((ZDat[i])-(ar0_))<=ar0x_)+ar0_*(abs((ZDat[i])-(ar0_))>ar0x_))
                            ZDat[i][0:Nf-NNew]=arr_z[0:Nf-NNew].copy()
                        
                        if hhh==0:
                            if Lo:
                                arr_rezBzz=np.exp(np.median(np.log(ZDat),axis=0)) 
                            else:
                                arr_rezBzz=np.median(ZDat,axis=0)
                        for i in range(anI):
                            if not astart0==np.Inf:
                                if Lo:                                
                                    ZDat[i][1:]=np.diff(np.log(ZDat[i])+KPP*np.log(arr_rezBzz))/(1+KPP)
                                else:
                                    ZDat[i][1:]=np.diff(ZDat[i]+KPP*arr_rezBzz)/(1+KPP)
                                ZDat[i][0]=0     
                            else:
                                if Lo:
                                    ZDat[i]=(np.log(ZDat[i])+KPP*np.log(arr_rezBzz))/(1+KPP)
                                else:
                                    ZDat[i]=(ZDat[i]+KPP*arr_rezBzz)/(1+KPP)
                
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
                            while hhhx<int(NIter):
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
                                                        
                                P=np.zeros(3,float)
                                PP=1
                                
                                seq0=(dd.reshape(len(dd)*len(dd[0])))[1:]*np.ceil(0.5*(1/(mdd4.reshape(len(dd)*len(dd[0]))==1)[0:len(dd)*len(dd[0])-1]+1/(mdd4.reshape(len(dd)*len(dd[0]))==1)[1:]))
                                seq0=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seq0)),float) 
                                seq0=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seq0)),float)    
                                
                                dd_AA=dd.copy()
                                dd_BB=dd.copy()
                                dd_CC=dd.copy()
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
                                      
                                        dd2=0.5*(dd_AA+dd_BB)[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]
                                        seqB=(dd2.reshape(len(dd2)*len(dd2[0])))[1:]*np.ceil(0.5*(1/(mdd4_.reshape(len(dd2)*len(dd2[0]))==1)[0:len(dd2)*len(dd2[0])-1]+1/(mdd4_.reshape(len(dd2)*len(dd2[0]))==1)[1:]))
                                        seqB=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqB)),float) 
                                        seqB=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqB)),float)    
                                    
                                        P[0:2]=np.polyfit(seqA,seqB,1)
                                        try:
                                            if not abs(P[0]-1)>0.5 and 100*scp.pearsonr(seqA,seqB)[0]>50:
                                                dd_CC[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=(dd2.copy()-P[1])/P[0] 
                                            else:
                                                if len(dd1)>1 and len(dd1[0])>=len(dd1):
                                                    eeA= (XFilter.RALF1FilterX( dd1*(1-seqA_)+seqA_*( dd1-(DD__B)),len(dd1),len(dd1[0]),1,0))
                                                    eeB=-(XFilter.RALF1FilterX(-dd1*(1-seqA_)+seqA_*(-dd1-(DD__A)),len(dd1),len(dd1[0]),1,0))
                                                    dd_AA[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=eeB.copy()#*(eeB>0)*((eeA+eeB)>0)
                                                    dd_BB[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=eeA.copy()#*(eeA<0)*((eeA+eeB)<0)
                                              
                                                dd2=0.5*(dd_AA+dd_BB)[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]
                                                seqB=(dd2.reshape(len(dd2)*len(dd2[0])))[1:]*np.ceil(0.5*(1/(mdd4_.reshape(len(dd2)*len(dd2[0]))==1)[0:len(dd2)*len(dd2[0])-1]+1/(mdd4_.reshape(len(dd2)*len(dd2[0]))==1)[1:]))
                                                seqB=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seqB)),float) 
                                                seqB=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seqB)),float)    
                                            
                                                P[0:2]=np.polyfit(seqA,seqB,1)
                                                try:
                                                    if not abs(P[0]-1)>0.5 and 100*scp.pearsonr(seqA,seqB)[0]>50:
                                                        dd_CC[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(jj*Nf/aMM):int((jj+1)*Nf/aMM)]=(dd2.copy()-P[1])/P[0] 
                                                    else:
                                                        PP=0
                                                except:
                                                    PP=0    

                                        except:
                                            PP=0    
                                            
                                seq0_=(dd_CC.reshape(len(dd_CC)*len(dd_CC[0])))[1:]*np.ceil(0.5*(1/(mdd4.reshape(len(dd_CC)*len(dd_CC[0]))==1)[0:len(dd_CC)*len(dd_CC[0])-1]+1/(mdd4.reshape(len(dd_CC)*len(dd_CC[0]))==1)[1:]))
                                seq0_=np.asarray(list(filter(lambda x: abs(x)!= np.Inf, seq0_)),float) 
                                seq0_=np.asarray(list(filter(lambda x: abs(np.isnan(x))!= 1, seq0_)),float)    
                                
                                P[0:2]=np.polyfit(seq0,seq0_,1)
                                if not abs(P[0]-1)>0.5 and 100*scp.pearsonr(seq0,seq0_)[0]>50:
                                    dd_CC=(dd_CC-P[1])/P[0] 
                                else:
                                    PP=0
                                if not PP==0:
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
                                        x=np.log(ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                                        y_1=filterFourierQ(aMx_,np.log(ar0_),NNew,1)[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                        y_2=filterFourierQ(aMn_,np.log(ar0_),NNew,1)[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()

                                    else:
                                        x=ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                        y_1=filterFourierQ(aMx_,(ar0_),NNew,1)[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                        y_2=filterFourierQ(aMn_,(ar0_),NNew,1)[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                    # if Lo:
                                    #     x=np.log(ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                                    #     y_1=aMx_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                    #     y_2=aMn_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()

                                    # else:
                                    #     x=ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                    #     y_1=aMx_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                    #     y_2=aMn_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                                    
                                    P_1=P.copy()
                                    P_2=P.copy()
                                    P_1[0:2]=np.polyfit(x,y_1,1)
                                    P_2[0:2]=np.polyfit(x,y_2,1)
                                    
                                    if not abs(P_1[0]-1)<0.5 or not abs(P_2[0]-1)<0.5 or 100*scp.pearsonr(x,y_1)[0]<0 or 100*scp.pearsonr(x,y_2)[0]<0:
                                        PP=0
                                if not PP==0:
                                    if Lo:
                                        arr_RezM[iGr][Nf-NNew:]=(filterFourierQ(aMx_,np.log(ar0_),NNew,1)[Nf-NNew:]-P_1[1])/P_1[0]
                                        arr_RezN[iGr][Nf-NNew:]=(filterFourierQ(aMn_,np.log(ar0_),NNew,1)[Nf-NNew:]-P_2[1])/P_2[0]
                                        arr_RezM[iGr][:Nf-NNew]=np.log(ar0_[:Nf-NNew])
                                        arr_RezN[iGr][:Nf-NNew]=np.log(ar0_[:Nf-NNew])
                                    else:
                                        arr_RezM[iGr][Nf-NNew:]=(filterFourierQ(aMx_,(ar0_),NNew,1)[Nf-NNew:]-P_1[1])/P_1[0]
                                        arr_RezN[iGr][Nf-NNew:]=(filterFourierQ(aMn_,(ar0_),NNew,1)[Nf-NNew:]-P_2[1])/P_2[0]
                                        arr_RezM[iGr][:Nf-NNew]=ar0_[:Nf-NNew].copy()
                                        arr_RezN[iGr][:Nf-NNew]=ar0_[:Nf-NNew].copy()
                                    # if Lo:
                                    #     arr_RezM[iGr][Nf-NNew:]=(aMx_[Nf-NNew:]-P_1[1])/P_1[0]
                                    #     arr_RezN[iGr][Nf-NNew:]=(aMn_[Nf-NNew:]-P_2[1])/P_2[0]
                                    #     arr_RezM[iGr][:Nf-NNew]=np.log(ar0_[:Nf-NNew])
                                    #     arr_RezN[iGr][:Nf-NNew]=np.log(ar0_[:Nf-NNew])
                                    # else:
                                    #     arr_RezM[iGr][Nf-NNew:]=(aMx_[Nf-NNew:]-P_1[1])/P_1[0]
                                    #     arr_RezN[iGr][Nf-NNew:]=(aMn_[Nf-NNew:]-P_2[1])/P_2[0]
                                    #     arr_RezM[iGr][:Nf-NNew]=ar0_[:Nf-NNew].copy()
                                    #     arr_RezN[iGr][:Nf-NNew]=ar0_[:Nf-NNew].copy()                                
                                    aMx0=aMx_.copy()
                                    aMn0=aMn_.copy()
                                                                                               
                                    if not PP==0:                                    
                                        if dNIt*int(hhhx/dNIt)==hhhx:
                                            all_RezM[iGr][hhh]=arr_RezM[iGr].copy()
                                            all_RezN[iGr][hhh]=arr_RezN[iGr].copy() 
                                        else:
                                            all_RezM[iGr][hhh]=np.maximum(all_RezM[iGr][hhh],arr_RezM[iGr])
                                            all_RezN[iGr][hhh]=np.minimum(all_RezN[iGr][hhh],arr_RezN[iGr])                                        
                                        if hhhx==0:
                                            dd1a[iGr,hhh]=(all_RezM[iGr][hhhx]).copy()#-P[1])/P[0]
                                            dd2a[iGr,hhh]=(all_RezN[iGr][hhhx]).copy()#-P[1])/P[0]
                                        else:
                                            dd1a[iGr,hhh]=(dd1a[iGr,hhh]*hhhx+(all_RezM[iGr][hhh]))/(hhhx+1)#-P[1])/P[0])/(hhhx+1)
                                            dd2a[iGr,hhh]=(dd2a[iGr,hhh]*hhhx+(all_RezN[iGr][hhh]))/(hhhx+1)#-P[1])/P[0])/(hhhx+1)                                
                                        hhhx=hhhx+1
                                    else:
                                        PP=0
                                if PP==0:
                                    aMx_=aMx0.copy()
                                    aMn_=aMn0.copy()
                                    aa=RandomQ(Nf)                        
                                    ss0=np.concatenate((aa, aa, aa))
                                tm1=tm.time()
                                if (tm1-tm0)>MxTime:
                                    break
                                            
                        tm1=tm.time()
                        if (tm1-tm0)>MxTime:
                            break
                        
                        dd1=np.amax(dd1a[iGr,max(0,(hhh+1)-int(dNIt/2+1)):hhh+1],axis=0)
                        dd2=np.amin(dd2a[iGr,max(0,(hhh+1)-int(dNIt/2+1)):hhh+1],axis=0)
                        
                        if Lo:
                            arr_RezM[iGr][Nf-NNew:]=dd1[Nf-NNew:]
                            arr_RezN[iGr][Nf-NNew:]=dd2[Nf-NNew:]
                        else: 
                            arr_RezM[iGr][Nf-NNew:]=dd1[Nf-NNew:]
                            arr_RezN[iGr][Nf-NNew:]=dd2[Nf-NNew:]
                      
                        if Lo:
                            x=np.log(ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                        else:
                            x=ar0_[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                        
                        y_1=arr_RezM[iGr][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                        y_2=arr_RezN[iGr][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                        P_1[0:2]=np.polyfit(x,y_1,1)                    
                        P_2[0:2]=np.polyfit(x,y_2,1)
                        # P[0]=np.std(y)/np.std(x)
                        # P[1]=np.mean(y)-P[0]*np.mean(x)
                        P_1[0]=np.std(y_1)/np.std(x)
                        P_1[1]=np.mean(y_1)-P_1[0]*np.mean(x)
                        P_2[0]=np.std(y_2)/np.std(x)
                        P_2[1]=np.mean(y_2)-P_2[0]*np.mean(x)
                        PP=(abs(P_1[0]-1)>1) or (abs(P_2[0]-1)>1)
   
                        all_RezNM[iGr][hhh][Nf-NNew:]=0.5*((arr_RezM[iGr][Nf-NNew:]-P_1[1])/P_1[0]
                                                           +(arr_RezN[iGr][Nf-NNew:]-P_2[1])/P_2[0])
                            
                        if not astart0==np.Inf:
                            all_RezMM[iGr][hhh]=np.cumsum(all_RezNM[iGr][hhh])
                        else:
                            all_RezMM[iGr][hhh]=all_RezNM[iGr][hhh].copy()
                        
                        # if Lo:
                        #     all_RezMM[iGr][hhh][Nf-NNew:]=(filterFourierQ((all_RezMM[iGr][hhh]),np.log(ar0x),NNew,1))[Nf-NNew:]
                        # else: 
                        #     all_RezMM[iGr][hhh][Nf-NNew:]=(filterFourierQ((all_RezMM[iGr][hhh]),(ar0x),NNew,1))[Nf-NNew:]
                        if Lo:
                            all_RezMM[iGr][hhh][Nf-NNew:]=all_RezMM[iGr][hhh][Nf-NNew:]
                        else: 
                            all_RezMM[iGr][hhh][Nf-NNew:]=all_RezMM[iGr][hhh][Nf-NNew:]
                        
                        if Lo:
                            x=np.log(ar0[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])
                        else:
                            x=ar0[Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()
                    
                        y=all_RezMM[iGr][hhh][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))].copy()                                       
                        
                        P[0:2]=np.polyfit(x,y,1)
                        # P[0]=np.std(y)/np.std(x)
                        # P[1]=np.mean(y)-P[0]*np.mean(x)
                        if PP or abs(P[0]-1)>1 or 100*scp.pearsonr(x,all_RezMM[iGr][hhh][Nf-NNew:Nf-NNew+int(lSrez*(NNew-(Nf-len(ar0))))])[0]<20:
                            MMM=MMM+1
                        all_RezMM[iGr][hhh][Nf-NNew:]=(all_RezMM[iGr][hhh][Nf-NNew:]-P[1])/P[0]
                        if Lo:
                            all_RezMM[iGr][hhh][:Nf-NNew]=np.log(ar0[:Nf-NNew])
                        else:
                            all_RezMM[iGr][hhh][:Nf-NNew]=ar0[:Nf-NNew].copy()
                        
                        arr_RezM[iGr]=(np.mean(all_RezMM[iGr][0:hhh+1],axis=0)+np.mean(all_RezMM[iGr][0:hhh+1],axis=0))/2
                    
                    tm1=tm.time()
                    if (tm1-tm0)>MxTime:
                        break
                    MMM=int(2*MMM/Ngroup)
                    arr_rezBz=np.mean(arr_RezM, axis=0) 
                    # arr_rezBz[1:]=np.diff(arr_rezBz)
                    # for iGr in range(Ngroup): 
                    #     arr_RezM[iGr][1:]=np.diff(arr_RezM[iGr])    
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
                
                tm1=tm.time()
                if MMM_<2*Nproc and np.std(mm1)>0 and np.std(mm2)>0 and (tm1-tm0)<MxTime:
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
                    if WrtTodr>0 or 33*int((hhh+1)/33)==(hhh+1) or hhha==(hhh+1):
                        out = cv.VideoWriter(wrkdir + aname+'.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
                        for icl in range(len(ImApp)):
                            cimgx=(cv.cvtColor(np.array(ImApp[icl]), cv.COLOR_RGB2BGR)) 
                            out.write(cimgx[0:gray_sz2,0:gray_sz1,:]) 
                        out.release()
                        del(out)
                    plt.show()
                    hhh=hhh+1
                    ##arr_z=arr_rezBzz.copy()
                    # if np.max(arr_RezM[:,Nf-NNew:])<1.2*arr_rezBz[Nf-NNew-1]:
                    #     hhh=int(NIter/33)
                    #     nnams__=nnams_[:len(nnams_)-1].copy()
                    #     nnams__[uuii:]=nnams_[uuii+1:].copy()
                    #     hkl.dump(nnams__,wrkdir + "name.rlf1")   
                    #     uuii=uuii-1
                    #     Nii=Nii-1
                    #     key=13
                    #     break
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

                    
                
