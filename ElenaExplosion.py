import concurrent.futures
import pylab as plt
import numpy as np
import pandas as pd
from PIL import Image 
import cv2 as cv
#import time as tm
import hickle as hkl
import os

from scipy import stats as scp
import dill 
#from scipy.signal import savgol_filter
import scipy.interpolate as interp

from RALf1FiltrVID import filterFourierQ
from RALf1FiltrVID import RALf1FiltrQ
from RALf1FiltrVID import RandomQ
import RALF1FilterX as XFilter
 

wrkdir = r"c:\Work\\W7\\"
aname='lena-Geo'
Lengt=1000
Ngroup=3
Nproc=2*Ngroup#*(mp.cpu_count())
Lo=0
aTmStop=6
NIt=3
NIter=100
DT=0.35
Nf_K=3
aDecm=1

def decimat(adat_):
    if Lo:
        adat_=np.log(adat_)
    adatx=0
    k=0
    adat__=np.zeros(int(len(adat_)/aDecm),float)
    for i in range(len(adat_)):
        adatx=adatx+adat_[i]
        if int(i/aDecm)*aDecm==i and i>0:
            adat__[k]=adatx/aDecm
            k=k+1
            adatx=0
    if Lo:
        return np.exp(adat__[1:len(adat__)-1])
    else:
        return (adat__[1:len(adat__)-1])

def fig2img ( fig ):
    fig.savefig(wrkdir +'dynamic.png',dpi=150,transparent=False,bbox_inches = 'tight')
    frame=Image.open(wrkdir +'dynamic.png')
    # fig.canvas.draw()
    # frame=Image.frombytes('RGB', fig.canvas.get_width_height(),
    #                        fig.canvas.tostring_rgb())
    return frame
    
def loaddata(aLengt,key=1):
    arrr=[]
    excel_data_df = pd.read_excel('lena.xls', sheet_name='1')
    
    dat=np.asarray(excel_data_df, float)
    f=interp.interp1d(dat[:,0], dat[:,1],'cubic')
    xnew=np.asarray(range(3,463),float)[::4]
    arrr=np.asarray(f(xnew),float)
    arrr=decimat(arrr)
            
    return arrr

if __name__ == '__main__':   
    ImApp=[]
        
    arrrxx=loaddata(Lengt,1)
    arrrxx=np.asarray(arrrxx,float)
 
    arr_rezDzRez=[]
    kkk=0
    out=0       
            
    w=0
  
    try:
        dill.load_session(wrkdir + aname+".ralf")
    except:
        aname=aname
        arrr=np.asarray(arrrxx).copy()  

        arrr=np.asarray(arrr,float)    
        Lengt=len(arrr)
        Nf=Lengt
        
        nn=int(Nf*DT)             
        NNew=int(Nf*0.6)  
        Nf=Nf+nn        
        ar0=np.asarray(arrr[0:])           
        
        arr_z=np.zeros(Nf,float)
        arr_z[0:Nf-NNew]=arrr[0:Nf-NNew].copy()
        arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]
          
        adat0=''
                               
        Arr_AAA=np.zeros((Ngroup,int(Nproc*NIter/Ngroup),Nf),float) 
        arr_rezBz=np.zeros(Nf,float)
        mn=np.mean(arr_z[0:Nf-NNew])
        
        all_rezAz=np.zeros((NIter,Nf),float)
        arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]  
        all_RezN=np.zeros((Ngroup,NIter,Nf),float)
        all_RezM=np.zeros((Ngroup,NIter,Nf),float)
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
            axes.plot(ar0, 'r.')
            axes.plot(arr_z, 'go-')  #cut data used for model
            axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s'%(aname),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=axes_.transAxes,color='blue', fontsize=14)        

            try:
                frame=fig2img(fig)  
            except:
                os.mkdir(wrkdir)
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
                mn=np.mean(arr_z[0:Nf-NNew])
                arr_rezMx=  np.zeros((Ngroup,Nf),float)
                arr_rezMn=  np.zeros((Ngroup,Nf),float)
                Arr_AAA=np.zeros((Ngroup,int(Nproc*NIter/Ngroup),Nf),float) 
                all_rezAz=np.zeros((NIter,Nf),float)
                arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]  
                all_RezN=np.zeros((Ngroup,NIter,Nf),float)
                all_RezM=np.zeros((Ngroup,NIter,Nf),float)
                all_RezNM=np.zeros((Ngroup,NIter,Nf),float)
                all_RezMM=np.zeros((Ngroup,NIter,Nf),float)
                hhh_=hhh_+1
            else:
                hhh_=hhh_+1
                ZZ=1
        
        if ZZ==0:                  
            try:
                [hhha,Arr_BBB]=(hkl.load(wrkdir + aname+".rlf1"))       
                Arr_AAA[:,0:int(Nproc*hhha/Ngroup),:Nf]=Arr_BBB[:,0:int(Nproc*hhha/Ngroup),:Nf].copy()
            except:            
                hhha=hhh-1
            
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

                with concurrent.futures.ThreadPoolExecutor(max_workers=Nproc) as executor:
                    future_to = {executor.submit(RALf1FiltrQ, argss[iProc]) for iProc in range(Nproc)}
                    for future in concurrent.futures.as_completed(future_to):                
                        arezAMx.append(future.result())
                del(future)                        
                del(executor)
                
                arezAMx= np.asarray(arezAMx,float)*Klg+Asr
                
                for iGr in range(Ngroup):
                    Arr_AAA[iGr][hhh*int(Nproc/Ngroup):(hhh+1)*int(Nproc/Ngroup)]=(
                        arezAMx[int(iGr*(Nproc/Ngroup)):int((iGr+1)*(Nproc/Ngroup))]).copy()
                
            WrtTodr=0
            if hhh>=hhha-1:    
                WrtTodr=1
                aDur=4
                    
            dNIt=NIter/3
            NQRandm=512
            aNN=2
            aMM=2
            nI=max(0,hhh-int(NIter/dNIt)+1)
            arr_RezM=  np.zeros((Ngroup,Nf),float)  
            for iGr in range(Ngroup):                
                ZDat_=Arr_AAA[iGr][((hhh)-nI)*int(Nproc/Ngroup):(hhh+1)*int(Nproc/Ngroup),:].copy()
                anI=len(ZDat_)
                ZDat=ZDat_.copy()
                aStr=np.Inf
                # aStr=ZDat[0][0]
                # for i in range(anI):
                #     ZDat[i,0]=0
                #     ZDat[i,1:]=np.diff(ZDat_[i,:])
                
                arr_RezM[iGr]=(np.mean(ZDat,axis = 0)+np.mean(ZDat,axis = 0))/2  
                hhhx=0
                all_RezM[iGr][hhhx]=arr_RezM[iGr].copy()
                all_RezN[iGr][hhhx]=arr_RezM[iGr].copy() 
                if anI>aNN:   
                    aMx=np.zeros(Nf,float)-np.Inf
                    aMn=np.zeros(Nf,float)+np.Inf
                    aMx_=0
                    aMn_=0
                    for hhhx in range(anI):
                        D=np.std(ZDat)                     
                        aa=RandomQ(Nf,NQRandm)                        
                        ss4=np.concatenate((aa, aa, aa))
                        DD=[]
                        for hhhc in range(anI):
                            DD.append(ss4[hhhc:hhhc+Nf])
                        DD=np.asarray(DD,float)                              
                        DD=(DD/np.std(DD)+1e-6)*D/2   

                        aa=RandomQ(Nf,NQRandm)                        
                        ss4_=np.concatenate((aa, aa, aa))
                        
                        DD_=[]
                        for hhhc in range(anI):
                            DD_.append(ss4_[hhhc:hhhc+Nf])
                        DD_=np.asarray(DD_,float)                              
                        DD_=(DD_/np.std(DD_))*D*2
                        DD__=(DD_-np.mean(DD_))                                      
                        
                        mn=np.mean(ZDat)
                        dd=(ZDat-mn)
                        
                        aa=RandomQ(Nf,NQRandm)                        
                        ss4=np.concatenate((aa, aa, aa))
                        liix=np.zeros((anI,Nf),int)

                        for i in range(anI):  
                            liix[i]=ss4[i:i+Nf].copy()
                            DD__[i,0:Nf-NNew]=0*DD_[i,0:Nf-NNew]
                            DD__[i,:]=(DD__[i,liix[i]])
                            dd[i]=(dd[i])[liix[i]].copy() 
                            
                        for ii in range(aNN):   
                            for jj in range(aMM):
                                dd1=dd[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(ii*Nf/aNN):int((ii+1)*Nf/aNN)]
                                ddA= dd1*(1-(dd1<0))
                                ddB=-dd1*(1-(dd1>0))
                                dA=DD[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(ii*Nf/aNN):int((ii+1)*Nf/aNN)]*(ddA==0)
                                dB=DD[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(ii*Nf/aNN):int((ii+1)*Nf/aNN)]*(ddB==0)
                                ddA=ddA+(DD__[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(ii*Nf/aNN):int((ii+1)*Nf/aNN)])
                                ddB=ddB+(DD__[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(ii*Nf/aNN):int((ii+1)*Nf/aNN)])
                                eeA=(XFilter.RALF1FilterX(  ddA-dA,len(ddA),len(ddA[0]),1,0)+
                                     XFilter.RALF1FilterX(  ddA-dB,len(ddA),len(ddA[0]),1,0))/2
                                eeB=-(XFilter.RALF1FilterX(  ddB-dB,len(ddB),len(ddB[0]),1,0)+
                                      XFilter.RALF1FilterX(  ddB-dA,len(ddB),len(ddB[0]),1,0))/2
                                dd[int(ii*anI/aNN):int((ii+1)*anI/aNN),int(ii*Nf/aNN):int((ii+1)*Nf/aNN)]=mn+(eeA*(1-(eeA>0))+eeB*(1-(eeB<0)))/2#(eeA+eeB)/2
                            
                        for i in range(anI):
                            aMx[liix[i]]=np.maximum(aMx[liix[i]],dd[i])
                            aMn[liix[i]]=np.minimum(aMn[liix[i]],dd[i])
                            aMx_=(aMx_*i+aMx)/(i+1)
                            aMn_=(aMn_*i+aMn)/(i+1)

                        all_RezM[iGr][hhhx]=(all_RezM[iGr][hhhx-1]*hhhx+np.maximum(all_RezM[iGr][hhhx-1],aMx_))/(hhhx+1)
                        all_RezN[iGr][hhhx]=(all_RezN[iGr][hhhx-1]*hhhx+np.minimum(all_RezN[iGr][hhhx-1],aMn_) )/(hhhx+1)
                                                              
                if Lo:
                    if aStr==np.Inf:
                        arr_RezM[iGr]=filterFourierQ(all_RezM[iGr][hhhx],np.log(arr_z),NNew,1)
                    else:                        
                        arr_RezM[iGr]=filterFourierQ(aStr+np.cumsum(all_RezM[iGr][hhhx]),np.log(arr_z),NNew,1)
                    arr_RezM[iGr][0:Nf-NNew]=np.log(ar0[0:Nf-NNew])  
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-np.mean(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                else:
                    if aStr==np.Inf:
                        arr_RezM[iGr]=filterFourierQ(all_RezM[iGr][hhhx],(arr_z),NNew,1)
                    else:
                        arr_RezM[iGr]=filterFourierQ(aStr+np.cumsum(all_RezM[iGr][hhhx]),(arr_z),NNew,1)
                    arr_RezM[iGr][0:Nf-NNew]=ar0[0:Nf-NNew].copy()
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-np.mean(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean((ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))

                all_RezMM[iGr][hhh]=(all_RezMM[iGr][hhh-1]*hhh+arr_RezM[iGr])/(hhh+1) 

                if Lo:
                    arr_RezM[iGr]=filterFourierQ(aStr+np.cumsum(all_RezN[iGr][hhhx]),np.log(arr_z),NNew,1)
                    arr_RezM[iGr][0:Nf-NNew]=np.log(ar0[0:Nf-NNew])  
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-np.mean(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                else:
                    arr_RezM[iGr]=filterFourierQ(aStr+np.cumsum(all_RezM[iGr][hhhx]),arr_z,NNew,1)
                    arr_RezM[iGr][0:Nf-NNew]=ar0[0:Nf-NNew].copy()
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-np.mean(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean((ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
    
                all_RezNM[iGr][hhh]=(all_RezNM[iGr][hhh-1]*hhh+arr_RezM[iGr])/(hhh+1)                
                arr_RezM[iGr]=(all_RezMM[iGr,hhh]+all_RezNM[iGr,hhh])/2
                                    
            arr_rezBz=(np.mean(arr_RezM, axis=0)+np.mean(arr_RezM, axis=0))/2  
            if Lo:
                arr_rezBz[Nf-NNew:]=(arr_rezBz[Nf-NNew:]/np.std(arr_rezBz[Nf-NNew:Nf-NNew+int(NNew*0.1)])) *np.std(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                arr_rezBz[Nf-NNew:]=(arr_rezBz[Nf-NNew:]-np.mean(arr_rezBz[Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                for iGr in range(Ngroup): 
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]/np.std(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) *np.std(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-np.mean(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
            else:
                arr_rezBz[Nf-NNew:]=(arr_rezBz[Nf-NNew:]/np.std(arr_rezBz[Nf-NNew:Nf-NNew+int(NNew*0.1)])) *np.std(np.log(ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                arr_rezBz[Nf-NNew:]=(arr_rezBz[Nf-NNew:]-np.mean(arr_rezBz[Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean((ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                for iGr in range(Ngroup): 
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]/np.std(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) *np.std((ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                    arr_RezM[iGr][Nf-NNew:]=(arr_RezM[iGr][Nf-NNew:]-np.mean(arr_RezM[iGr][Nf-NNew:Nf-NNew+int(NNew*0.1)])) +np.mean((ar0[Nf-NNew:Nf-NNew+int(NNew*0.1)]))
                                                        
            if Lo:   
                for iGr in range(Ngroup):    
                    arr_RezM[iGr]=np.exp(arr_RezM[iGr]) 
                arr_rezBz=np.exp(arr_rezBz) 
                
            mm1=ar0[Nf-NNew:].copy()                            
            mm2=arr_rezBz[Nf-NNew:len(ar0)].copy()   
            if np.std(mm1)>0 and np.std(mm2)>0:
                if hhh>=hhha:                               
                    hkl.dump([hhh+1,Arr_AAA], wrkdir + aname+".rlf1")                        
                Koef_.append(100*scp.pearsonr(mm1,mm2)[0])                               
                fig = plt.figure()
                axes = fig.add_axes([0.1, 0.1, 1.2, 1.2])
                axes_ = fig.add_axes([0, 0, 0.3, 0.3]) 
                axes__ = fig.add_axes([0.4, 0, 0.3, 0.3])                     
                axes.plot(ar0, 'ro-', alpha=0.1)
                axes.plot(arrr, 'rx-')
                for iGr in range(Ngroup):
                    axes.plot(arr_RezM[iGr],linewidth=3.,alpha=0.2)
                axes.plot(arr_rezBz,'yx-',linewidth=4.,alpha=0.5)
                axes.text(-0.1, 4, '%s  '%(hhh+1),
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=axes_.transAxes,color='black', fontsize=24) 
                axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s'%(aname),
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axes_.transAxes,color='blue', fontsize=14)     
                gint=np.polyfit(mm1,mm2,1)
                axes_.plot(mm1,gint[1]+gint[0]*mm1,'y.',linewidth=2.)                    
                axes_.plot(mm1,mm2, 'ok', markersize=3, alpha=0.1) 
                axes_.text(0.2, 0.6, '%d'%int(np.floor(np.asarray(Koef_,float)[::-1][0])),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=axes_.transAxes,color='green', fontsize=30)
                
                axes__.text(1.8, 0.6, 'Dunning –Kruger\n effect',
                        verticalalignment='bottom', horizontalalignment='center',
                    transform=axes_.transAxes,color='green', fontsize=14)  
                axes__.plot(Koef_,'y',linewidth=2.)
                
                frame=fig2img(fig)  
                cimg = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)        
                gray_sz1=min(gray_sz1,len(cimg[0]))
                gray_sz2=min(gray_sz2,len(cimg))
                ImApp.append(frame)
                if WrtTodr>0:
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
                dill.dump_session(wrkdir + aname+".ralf")   
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
    axes.plot(arrr, 'ro-')
    axes.plot(arr_rezBz, 'cx-', alpha=0.5) #predicted data
    axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s'%(aname),
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes_.transAxes,color='blue', fontsize=14)     
    gint=np.polyfit(mm1,mm2,1)
    axes_.plot(mm1,gint[0]+gint[1]*mm1,'y',linewidth=2.) 
    axes_.plot(mm1,mm2, 'ok', markersize=3, alpha=0.1)    
    axes_.text(0.2, 0.6, '%d'%int(np.floor(np.asarray(Koef_,float)[::-1][0])),
        verticalalignment='bottom', horizontalalignment='right',
        transform=axes_.transAxes,color='green', fontsize=30)    
    
    axes__.text(1.8, 0.6, 'Dunning –Kruger\n effect',
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
