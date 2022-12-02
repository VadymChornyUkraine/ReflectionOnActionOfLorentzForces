import numpy as np
import cv2
from scipy import ndimage
import concurrent.futures
import multiprocessing as mp
from scipy.signal import savgol_filter
import dill 
from RALf1FiltrVID import RALf1FiltrQ,RandomQ,filterFourierQ
    
wrkdir = r".\\"
wwrkdir_=r".\W8\\"
nama='ZEISSAXIO_HeLa'
nmfile0=nama+'.mp4'
nmfile=nama+'out.mp4'
filename = wwrkdir_+"globalsavepkl"
   
if __name__ == '__main__':        
    anamef="fralf.tmp"
    fo = open(anamef, "w")
    fo.write(str(0)+'\n')
    fo.close()  
    hhh=0
    try:
        dill.load_session(filename+".ralf")  
    except:
        Ngroup=3
        Nproc=int(np.floor(mp.cpu_count()))-1
        coef=0.05#0.08
        astep=1
        NIt=9
        dNew=0.33
        NumSteps=4
        NCircls=8
        
        NChan=40#112
            
        # Create a VideoCapture object and read from input file 
        cap = cv2.VideoCapture(wwrkdir_ +nmfile0)#or mp4     
        ArrX=[[] for i in range(3)]
        aDur=int(cap.get(cv2.CAP_PROP_FPS))
        sz1=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sz2=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gray_sz1=sz1
        gray_sz2=sz2
        NumFr_=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        agray=[[] for icl in range(3)]
        ret=True
        kk=0
        frame_0=[]
        while kk<NumFr_ and ret:
            for ii in range(astep):
                ret, frame = cap.read()   
                if ret:
                    frame_0.append(frame.copy())
                    frame_=frame
                    frame0=frame
                    for icl in range(3):    
                        gray=frame[:,:,icl]
                        gray = ndimage.zoom(gray, coef)  
                        sz1=len(gray)
                        sz2=len(gray[0])
                        coefX=max(gray_sz1/sz1,gray_sz2/sz2)
                        if ii==0:
                            agray[icl]=gray                        
                        agray[icl]=(agray[icl]*ii+gray)/(ii+1)                    
                    if ii==astep-1:
                        frame= np.zeros((len(gray),len(gray[0]),3),np.uint8)           
                        for icl in range(3):
                            frame[ : , : , icl] =agray[icl]
                            ArrX[icl].append(agray[icl])
                            frame0[ : , : , icl] = ndimage.zoom(frame[ : , : , icl], coefX)[0:gray_sz1,0:gray_sz2]                       
                        cv2.imshow('frame', frame0)
                        # Press Q on keyboard to  exit 
                        if cv2.waitKey(30) & 0xFF == ord('q'): 
                            break   
            if ret:
                kk=kk+astep
            else:
                break            
        NumFr_=kk
        cap.release() 
        cv2.destroyAllWindows() 
        Arr=np.asarray(ArrX,float)    
        dill.dump_session(filename+".ralf")  
    
    ahh=0
    while hhh<NumSteps:
        hh=0
        NumFr0=len(ArrX[0])        
        while hh<NCircls: 
            try:
                dill.load_session(filename+("%s_%s"%(hhh,hh))+".ralf")    
                hh=ahh
            except:
                if hh==0:
                    NNew=int(NumFr0*dNew)
                    NumFr=NumFr0+int(np.ceil(hhh*NNew/NumSteps))
                    Nn0=NumFr-NNew+int(np.ceil(NNew/NumSteps))
                    ArrRezRez=np.zeros((3,NCircls,NumFr,sz1,sz2),float)  
                    ArrRez_=np.zeros((3,NumFr,sz1,sz2),float)                  
                    ArrRezMx=np.zeros((3,Ngroup,NumFr,sz1,sz2),float)-np.Inf
                    ArrRezMn=np.zeros((3,Ngroup,NumFr,sz1,sz2),float)+np.Inf 
                else:
                    dill.load_session(filename+("%s_%s"%(hhh,hh-1))+".ralf")                 
                    hh=ahh
                                         
                SZ=int(sz1*sz2)
                NumFri=RandomQ(SZ) 
                NumFrX_=np.zeros(SZ,int)
                NumFrY_=np.zeros(SZ,int)
                for i in range(SZ):
                    NumFrY_[i]= np.floor(NumFri[i]/sz1)
                    NumFrX_[i]=NumFri[i]-NumFrY_[i]*sz1 
                Ndel=int(np.ceil(SZ/NChan))
                SZ=NChan*Ndel
                Arr_=np.zeros((3,SZ,int(NumFr)),float)
                
                for kk in range(int(SZ/(sz1*sz2))+1):
                    if kk==0:
                        NumFrX= NumFrX_.copy()
                        NumFrY=NumFrY_.copy()  
                    else:
                        NumFrX=np.concatenate((NumFrX, NumFrX_))
                        NumFrY=np.concatenate((NumFrY, NumFrY_))    
                for icl in range(3):    
                    for i in range(int(NumFr-NNew)):
                        for j in range(SZ):
                            Arr_[icl][j][i]=Arr[icl][i][NumFrX[j]][NumFrY[j]]            
                    
                for l in range(Ndel):         
                    Arr_x=np.zeros((3,NChan,NumFr),float)
                    argss=[[] for kk in range(3*Nproc*Ngroup)]
                    for kk in range(3*Nproc*Ngroup):
                        argss[kk]=[0, "%s"%NChan, "%s"%NNew, "%s"%NIt]
                        icl=int(kk/Nproc/Ngroup)
                        Arr_x[icl]=Arr_[icl][l*NChan:(l+1)*NChan].copy()    
                        for i in range(NChan): 
                            for j in range(NumFr):
                                if j>NumFr-NNew-1:
                                    argss[kk].append(Arr_x[icl][i][NumFr-NNew-1])
                                else:
                                    argss[kk].append(Arr_x[icl][i][j])                                                       
                            
                    arezAMx=[]
                    # for kk in range(len(argss)):
                    #     dd=RALf1FiltrQ(argss[kk])
                    #     arezAMx.append(dd)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=Nproc) as executor:
                        future_to = {executor.submit(RALf1FiltrQ, argss[kk]) for kk in range(len(argss))}
                        for future in concurrent.futures.as_completed(future_to):                
                            arezAMx.append(future.result())
                        del(executor)            
                    
                    arezAMx=np.asarray(arezAMx,float)
                    for icl in range(3):   
                        arezAMx_=arezAMx[0+icl*Nproc*Ngroup:(icl+1)*Nproc*Ngroup].copy()
                        for ig in range(Ngroup):
                            for j in range(l*NChan,(l+1)*NChan):
                                dd=arezAMx_[:,(j-l*NChan)*NumFr+np.asarray(np.linspace(0,NumFr-1,NumFr),int)]
                                ArrRezMx[icl,ig,:,NumFrX[j],NumFrY[j]]=(np.maximum(ArrRezMx[icl,ig,:,NumFrX[j],NumFrY[j]],
                                                                                  np.max(dd[np.asarray(np.linspace(ig*Nproc,ig*Nproc+Nproc-1,Nproc),int),:],0)))
                                ArrRezMn[icl,ig,:,NumFrX[j],NumFrY[j]]=(np.minimum(ArrRezMn[icl,ig,:,NumFrX[j],NumFrY[j]],
                                                                                  np.min(dd[np.asarray(np.linspace(ig*Nproc,ig*Nproc+Nproc-1,Nproc),int),:],0)))                                  
                    
                    print ("calculated %s percents"%(int((l+1)/Ndel*1000)/10))
            #if hh==3:               
                for icl in range(3): 
                    ZZZ0=(np.max(ArrRezMx[icl],0)+np.min(ArrRezMn[icl],0))/2             
                    ZZZ0[0:NumFr-NNew]=Arr[icl,0:NumFr-NNew]
                    for i in range(len(ZZZ0[0])):
                        for j in range(len(ZZZ0[0][0])):
                            ZZZ0x=ZZZ0[:,i,j].copy()  
                            if i==0 and j==0:
                                ZZZ0_=ZZZ0x.copy()  
                            else:
                                ZZZ0_=np.concatenate((ZZZ0_,ZZZ0x))
                    ZZZ0_=filterFourierQ(ZZZ0_,ZZZ0_,NNew,len(ZZZ0[0])*len(ZZZ0[0][0]))    
                    for i in range(len(ZZZ0[0])):
                        for j in range(len(ZZZ0[0][0])):
                            ZZZ0[:,i,j]=ZZZ0_[0+NumFr*(i*len(ZZZ0[0][0])+j):NumFr+NumFr*(i*len(ZZZ0[0][0])+j)].copy()  
                        
                    ZZZZ=ZZZ0.copy()   
                    ArrRezRez[icl,hh]=ZZZZ.copy()
                    ZZZZ[NumFr-NNew:]=np.mean(ArrRezRez[icl,0:hh+1],0)[NumFr-NNew:]
                    
                    # affZZ=ZZZZ[NumFr-NNew:].copy()-np.Inf
                    # az=int(np.floor(NumFr-NNew)/NNew)
                    # for i in range(az):
                    #     ffZZ=np.fft.fftn(ZZZZ[0+NumFr-2*NNew-i*NNew:NumFr-NNew-i*NNew,:,:])
                    #     affZZ=np.maximum(affZZ,np.asarray(np.abs(ffZZ),float))
                    # ZZ_=np.fft.fftn(ZZZZ[NumFr-NNew:NumFr,:,:])
                    # aZZ_=np.asarray(np.abs(ZZ_),float)
                    # mZZ_=0.62*np.mean(aZZ_)
                    # dd=(ZZ_/aZZ_)*(1*(aZZ_>mZZ_))*affZZ
                    # ddd=np.zeros((2*len(dd),2*len(dd[0]),2*len(dd[0][0])),complex)
                    # ddd[0:2*len(dd):2,0:2*len(dd[0]):2,0:2*len(dd[0][0]):2]=dd.copy()
                    # ddd[0,0:2*len(dd[0]):2,0:2*len(dd[0][0]):2]=ZZ_[0,:,:]                    
                    # ddd[0:2*len(dd):2,0,0:2*len(dd[0][0]):2]=ZZ_[:,0,:]
                    # ddd[0:2*len(dd):2,0:2*len(dd[0]):2,0]=ZZ_[:,:,0]
                    # fZZ=np.fft.ifftn(ddd)#*(affZZ>maffZZ))
                    # ZZZZ[NumFr-NNew:]=fZZ.real[0:NNew,0:len(ZZZZ[0]),0:len(ZZZZ[0][0])]
                    
                    ZZZZ[NumFr-NNew:]=(np.std(Arr[icl,NumFr-NNew-5:NumFr-NNew-2])/np.std(ZZZZ[NumFr-NNew:NumFr-NNew+3]))*ZZZZ[NumFr-NNew:] 
                    ZZZZ[NumFr-NNew:]=ZZZZ[NumFr-NNew:]-np.mean(ZZZZ[NumFr-NNew])+2*np.mean(Arr[icl,NumFr-NNew-1])-np.mean(Arr[icl,NumFr-NNew-2])                 
                    ArrRez_[icl]=ZZZZ.copy()            
                                       
                ArrRez_=np.asarray(ArrRez_-(ArrRez_-255)*(ArrRez_>255),np.uint8)
                ArrRez_=np.asarray(ArrRez_-ArrRez_*(ArrRez_<0),np.uint8)
                coefX=max(gray_sz1/sz1,gray_sz2/sz2)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                for icl in range(3):
                    for i in range(sz1):
                        for j in range(sz2):                
                            ArrRez_[icl,:,i,j]= savgol_filter(ArrRez_[icl,:,i,j], 11, 5)
                            
                out = cv2.VideoWriter(wwrkdir_ +nmfile,fourcc, aDur, (gray_sz2,gray_sz1))
                kk=np.zeros(3,int)
                kkk=np.zeros(3,int)
                kk[icl]=0
                kkk[icl]=0
                for kk in range(NumFr-1):
                    frame=frame_
                    for ii in range(astep):                
                        for icl in range(3):   
                            frame[ : , : , icl] = ndimage.zoom(ArrRez_[icl][kk], coefX)[0:gray_sz1,0:gray_sz2]                       
                        #frame=cv2.medianBlur(frame,15)#int(3*coefX)) 
                        cv2.addWeighted(frame,0.8,frame_0[kk*astep+ii],0.2,0,frame)
                        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.imshow('frame', frame)
                        out.write(frame) 
                    # Press Q on keyboard to  exit 
                    if cv2.waitKey(30) & 0xFF == ord('q'): 
                        break                    
                out.release()
                cv2.destroyAllWindows()
                
                ahh=hh+1                
                dill.dump_session(filename+("%s_%s"%(hhh,hh))+".ralf")  
                hh=hh+1
                
       
        cap = cv2.VideoCapture(wwrkdir_ +nmfile)#or mp4      
        ArrXY=[[] for i in range(3)]    
        sz1=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sz2=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gray_sz1=sz1
        gray_sz2=sz2
        NumFr_=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        agray=[[] for icl in range(3)]
        ret=True
        kk=0
        frame_0=[]
        while kk<NumFr_ and ret:
            for ii in range(astep):
                ret, frame = cap.read()   
                if ret:
                    frame_0.append(frame.copy())
                    frame_=frame
                    for icl in range(3):    
                        gray=frame[:,:,icl]
                        gray = ndimage.zoom(gray, coef)  
                        if ii==0:
                            agray[icl]=gray                        
                        agray[icl]=(agray[icl]*ii+gray)/(ii+1)                    
                    if ii==astep-1:
                        frame= np.zeros((len(gray),len(gray[0]),3),np.uint8)           
                        for icl in range(3):
                            frame[ : , : , icl] =agray[icl]
                            ArrXY[icl].append(agray[icl])
                        cv2.imshow('frame', frame)
                        # Press Q on keyboard to  exit 
                        if cv2.waitKey(30) & 0xFF == ord('q'): 
                            break   
            if ret:
                kk=kk+astep
            else:
                break             
        cap.release() 
        cv2.destroyAllWindows()
        Arr=np.asarray(ArrXY,float)        
        sz1=len(gray)
        sz2=len(gray[0])
        hhh=hhh+1
        dill.dump_session(filename+".ralf")  