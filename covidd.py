import covid19 as COVID19Py
import pandas as pd
import numpy as np
import pylab as plt
import shapefile as shp
import seaborn as sns
import cv2 as cv
from PIL import Image
import multiprocessing as mp
import concurrent.futures
import dill
from scipy.signal import savgol_filter
#import numpy.matlib as mtlb

from RALf1FiltrVID import RALf1FiltrQ,RandomQ,filterFourierQ

wrkdir=".\W9\\"

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    sha=sf.shapes() 
    fields = [x[0] for x in sf.fields][1:]
    records=[]
    shps=[]
    NN=len(sha)
    fl=np.zeros(NN,int)
    for i in range(NN):
        try:
            records.append(sf.record(i))
            shps.append(sha[i].points)
            fl[i]=1
        except:
            records.append([])
            shps.append([])
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df, fl

def prep_map(sf,fl,figsize = (10,8)):
    fig=plt.figure(figsize = figsize)
    plt.style.use(['dark_background'])
    ax = plt.axes()
    ax.set_aspect('equal')   
    l=0
    for j in range(len(fl)):
        if fl[j]>0:
            shape_ex = sf.shape(j)
            x_lon = np.zeros((len(shape_ex.points),1))
            y_lat = np.zeros((len(shape_ex.points),1))
            for ip in range(len(shape_ex.points)):
                x_lon[ip] = shape_ex.points[ip][0]
                y_lat[ip] = shape_ex.points[ip][1]  
            if l==0:
                xlon=x_lon.copy()
                ylat=y_lat.copy()
                l=1
            else:
                xlon=np.concatenate((xlon,x_lon))
                ylat=np.concatenate((ylat,y_lat))
   
    return fig,ax,xlon,ylat    

def plotmap(sf,fl,xlon,ylat, x_lim = None, y_lim = None, figsize = (11,9)):
    fig=plt.figure(figsize = figsize)
    plt.style.use(['dark_background'])
    ax = plt.axes()
    ax.set_aspect('equal')           
    plt.plot(xlon,ylat,'.b',alpha=0.1) 

    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    return fig,ax

def fig2img ( fig ):
    fig.canvas.draw()
    return Image.frombytes('RGB', fig.canvas.get_width_height(),
                           fig.canvas.tostring_rgb())

def getCases():
    dVal=[]
    covid19 = COVID19Py.COVID19()
    data0=covid19.getAll(timelines=True)["locations"]
    NN=len(data0)
    x=np.zeros(NN,float)
    y=np.zeros(NN,float)
    data0UA=[]
    for i in range(NN):        
        data_=data0[i] 
        if data0[i]["country"]=="Ukraine":
            data0UA.append(data0[i])
        
        x[i]=data_["coordinates"]["longitude"]
        y[i]=data_["coordinates"]["latitude"]
        data1=data_["timelines"]["confirmed"]["timeline"]
        data2=data_["timelines"]["deaths"]["timeline"]
        values1= np.asarray(pd.Series(data1),int)
        values2= np.asarray(pd.Series(data2),int)
        Num=len(values1)
        val1=values1[1:]-values1[0:Num-1]
        val2=values2[1:]-values2[0:Num-1]    
        dVal.append(100*val2/(val1+1e-9)*(val1>0))
        #dVal.append(val2*(val2>0))
        
    dVal=np.asarray(dVal,float)
    dVal=dVal.transpose()
    return x,y,dVal

if __name__ == '__main__':   
    aname="COVID-19"

    sns.set(style="darkgrid", palette="bright", color_codes=True)
    sns.mpl.rc("figure", figsize=(10,8))
    shp_path = wrkdir+"TM_WORLD_BORDERS-0.3.shp"
    try:
        x,y,dVal=getCases()
    except:
        dill.load_session(wrkdir+aname+("%s_%s"%(0,0))+".ralf")    
        hh=0
        
    NN=len(dVal[0])
    MM=len(dVal)
    
    dt=0.1
    Mx=np.mean(dVal)+np.std(dVal)*4
    
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    aDur=4
    w=0
    dVal=((dVal)/Mx)*(dVal>0)
    dVal=dVal-(dVal>1)*(dVal-1)
    
    NumSteps=6#12
    NCircls=8#8

    NChan=40
    Ngroup=3
    Nproc=int(np.floor(mp.cpu_count()))-1

    NIt=9
    hhh=0
    while hhh<NumSteps:
        hh=0   
        ahh=0
        NumFr0=MM+int(hhh*(MM/4))
        NNew=int(NumFr0/3)#6
        while hh<NCircls:
            try:
                dill.load_session(wrkdir+aname+("%s_%s"%(hhh,hh))+".ralf")    
                hh=ahh
            except:
                if hh==0:
                    NumFr=NumFr0+NNew
                    ArrRezRez=np.zeros((NCircls,NumFr,NN),float)  
                    ArrRez_=np.zeros((NumFr,NN),float)                  
                    ArrRezMx=np.zeros((Ngroup,NumFr,NN),float)-np.Inf
                    ArrRezMn=np.zeros((Ngroup,NumFr,NN),float)+np.Inf 
                else:
                    dill.load_session(wrkdir+aname+("%s_%s"%(hhh,hh-1))+".ralf")                 
                    hh=ahh
                SZ=NN
                NumFri=RandomQ(NN)
                NumFri=np.concatenate((NumFri, NumFri))
                Ndel=int(np.ceil(SZ/NChan))
                SZ=NChan*Ndel
                Arr_=np.zeros((SZ,int(NumFr)),float)
                for i in range(int(NumFr-NNew)):
                    for j in range(SZ):
                        Arr_[j][i]=dVal[i][NumFri[j]]             
                            
                for l in range(Ndel):         
                    argss=[[] for kk in range(Nproc*Ngroup)]
                    for kk in range(len(argss)):
                        argss[kk]=[0, "%s"%NChan, "%s"%NNew, "%s"%NIt]
                        Arr_x=Arr_[l*NChan:(l+1)*NChan].copy()    
                        for i in range(NChan): 
                            for j in range(NumFr):
                                if j>NumFr-NNew-1:
                                    argss[kk].append(Arr_x[i][NumFr-NNew-1])
                                else:
                                    argss[kk].append(Arr_x[i][j])                                                  
                                    
                    arezAMx=[]
                    # for kk in range(len(argss)):
                    #     dd=RALf1FiltrQ(argss[kk])
                    #     arezAMx.append(dd)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=Nproc) as executor:
                        future_to = {executor.submit(RALf1FiltrQ, argss[kk]) for kk in range(len(argss))}
                        for future in concurrent.futures.as_completed(future_to):                
                            arezAMx.append(future.result())
                        del(executor)            
                            
                    arezAMx_=np.asarray(arezAMx,float)
                    for ig in range(Ngroup):
                        for j in range(l*NChan,(l+1)*NChan):
                            dd=arezAMx_[:,(j-l*NChan)*NumFr+np.asarray(np.linspace(0,NumFr-1,NumFr),int)]
                            ArrRezMx[ig,:,NumFri[j]]=(
                                np.maximum(ArrRezMx[ig,:,NumFri[j]],
                                                    np.max(dd[np.asarray(np.linspace(ig*Nproc,ig*Nproc+Nproc-1,Nproc),int),:],0)))
                            ArrRezMn[ig,:,NumFri[j]]=(
                                np.minimum(ArrRezMn[ig,:,NumFri[j]],
                                                    np.min(dd[np.asarray(np.linspace(ig*Nproc,ig*Nproc+Nproc-1,Nproc),int),:],0)))                                  
                            
                    print ("calculated %s percents"%(int((l+1)/Ndel*1000)/10))           
            #if hh==1:        
                ZZZ0=(np.max(ArrRezMx,0)+np.min(ArrRezMn,0))/2       
                ZZZ0[0:NumFr-NNew]=dVal[0:NumFr-NNew] 
                for i in range(NN):
                    ZZZ0x=ZZZ0[:,i].copy()  
                    if i==0:
                        ZZZ0_=ZZZ0x.copy()  
                    else:
                        ZZZ0_=np.concatenate((ZZZ0_,ZZZ0x))
                ZZZ0_=filterFourierQ(ZZZ0_,ZZZ0_,NNew,NN) 
                ZZZ0_=ZZZ0_-(ZZZ0_<0)*ZZZ0_       
                for i in range(NN):
                    ZZZ0[:,i]=ZZZ0_[0+NumFr*i:NumFr+NumFr*i].copy()  
                    
                ArrRez_M=ZZZ0.copy()
                ArrRezRez[hh]=ZZZ0.copy()
                ArrRez_M[NumFr-NNew:]=np.mean(ArrRezRez[0:hh+1],0)[NumFr-NNew:]
                ArrRez_M[NumFr-NNew:]=(np.std(ArrRez_M[NumFr-NNew-5:NumFr-NNew-2])/np.std(ArrRez_M[NumFr-NNew:NumFr-NNew+3]))*ArrRez_M[NumFr-NNew:]
                ArrRez_M[NumFr-NNew:]=ArrRez_M[NumFr-NNew:]-np.mean(ArrRez_M[NumFr-NNew])+2*np.mean(ArrRez_M[NumFr-NNew-1])-np.mean(ArrRez_M[NumFr-NNew-2])           
                dVal=ArrRez_M.copy()
                ArrRez_=ArrRez_M.copy()
                        
                ArrRez_=ArrRez_-(ArrRez_>1)*(ArrRez_-1)
                sf = shp.Reader(shp_path)
                df,fl = read_shapefile(sf)
                fig,axs,xlon,ylat=prep_map(sf,fl)
                ImApp=[]
                for j in range(len(ArrRez_)):                   
                    fig,axs=plotmap(sf,fl,xlon,ylat)
                    for i in range(NN):                    
                        dl=(20*ArrRez_[j][i])
                        if dl>0:
                            axs.plot((x[i]-dt,x[i]+dt),(y[i]-dt,y[i]+dt),'r',linewidth=dl)
                    
                    axs.text(0.2, 0.2, '%d'%int(j-MM+1),
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs.transAxes,color='green', fontsize=50)
                
                    frame=fig2img(fig)  
                    cimg = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)        
                    
                    if w==0:
                        gray_sz1=len(cimg[0])
                        gray_sz2=len(cimg)
                        w=1
                    else:
                        gray_sz1=min(gray_sz1,len(cimg[0]))
                        gray_sz2=min(gray_sz2,len(cimg))
                
                    plt.show()
                    
                    if j>0:
                        ImApp.append(frame)
                             
                del(sf)
                del(df)
                                
                if j>0:
                    out = cv.VideoWriter(wrkdir + aname+'.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
                    for icl in range(len(ImApp)-2):
                        cimgx=(cv.cvtColor(np.array(ImApp[icl]), cv.COLOR_RGB2BGR)) 
                        out.write(cimgx[0:gray_sz2,0:gray_sz1,:]) 
                    out.release() 
                
                ahh=hh+1
                dill.dump_session(wrkdir+aname+("%s_%s"%(hhh,hh))+".ralf") 
                hh=hh+1
        hhh=hhh+1
        