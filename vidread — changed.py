import numpy as np
import cv2
from scipy import ndimage
import concurrent.futures
import multiprocessing as mp
from scipy.signal import savgol_filter
from RALf1FiltrVID import RALf1FiltrV,RandomV,filterFourierV
import dill 

wrkdir = r".\\"
wwrkdir_=r".\W8\\"
nama='novalSARSCOV2'
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
        hhh=hhh        
        coef=0.033
        astep=10
        NIt=2
        dNew=0.4
        NumSteps=4
        NCircls=10
        Nproc=int(np.floor(mp.cpu_count()/3))
        Limite=60000
            
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
        while kk<NumFr_ and ret:
            for ii in range(astep):
                ret, frame = cap.read()   
                if ret:
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
    
    while hhh<NumSteps:
        hh=0
        NumFr0=len(ArrX[0])
        while hh<NCircls: 
            try:
                dill.load_session(filename+("%s_%s"%(hhh,hh))+".ralf")        
            except:                
                NNew=int(NumFr0*dNew)
                NumFr=NumFr0+int(np.ceil(hhh*NNew/NumSteps))
                Nn0=NumFr-NNew+int(np.ceil(NNew/NumSteps)) 
                if hh==0:
                    ArrRezMx=np.zeros((3,NumFr,sz1,sz2),float)-np.Inf
                    ArrRezMn=np.zeros((3,NumFr,sz1,sz2),float)+np.Inf                          
                SZ=int(sz1*sz2)
                NumFri=RandomV(SZ) 
                NumFrX_=np.zeros(SZ,int)
                NumFrY_=np.zeros(SZ,int)
                for i in range(SZ):
                    NumFrY_[i]= np.floor(NumFri[i]/sz1)
                    NumFrX_[i]=NumFri[i]-NumFrY_[i]*sz1 
                Ndel=(SZ*NumFr/Limite)
                NChan=int(np.ceil(sz1/Ndel))     
                SZ=NChan*int(np.ceil(sz2*Ndel))
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
                    
                for l in range(int(np.ceil(sz2*Ndel))):         
                    Arr_x=np.zeros((3,NChan,NumFr),float)
                    argss=[[] for kk in range(3*Nproc)]
                    for kk in range(3*Nproc):
                        argss[kk]=[0, "%s"%NChan, "%s"%NNew, "%s"%NIt]
                        icl=int(kk/Nproc)
                        Arr_x[icl]=Arr_[icl][l*NChan:(l+1)*NChan].copy()    
                        for i in range(NChan): 
                            for j in range(NumFr):
                                if j>NumFr-NNew-1:
                                    argss[kk].append(0)
                                else:
                                    argss[kk].append(Arr_x[icl][i][j]-Arr_x[icl][i][NumFr-NNew-1])                                                       
                            
                    arezAMx=[]
                    # for kk in range(3*Nproc):
                    #     dd=RALf1FiltrV(argss[kk])
                    #     arezAMx.append(dd)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=Nproc) as executor:
                        future_to = {executor.submit(RALf1FiltrV, argss[kk]) for kk in range(3*Nproc)}
                        for future in concurrent.futures.as_completed(future_to):                
                            arezAMx.append(future.result())
                        del(executor)            
                    
                    arezAMx=np.asarray(arezAMx,float)
                    for icl in range(3):   
                        arezAMx_=arezAMx[0+icl*Nproc:Nproc+icl*Nproc].copy()
                        arezBMx=np.zeros((NChan,NumFr),float)            
                        arezAMxZ=(np.max(arezAMx_,axis=0)+np.min(arezAMx_,axis=0))/2
                        
                        arezAMxZ=filterFourierV(arezAMxZ,np.reshape(Arr_x[icl],len(Arr_x[icl])*len(Arr_x[icl][0])),NNew,NChan)
                        
                        for i in range(NChan):
                            arezBMx[i]=arezAMxZ[0+NumFr*i:NumFr+NumFr*i].copy()              
                            # arezBMx[i][NumFr-NNew:NumFr]=arezBMx[i][NumFr-NNew:NumFr]*np.std(Arr_x[icl][i][NumFr-NNew:Nn0])/np.std(arezBMx[i][NumFr-NNew:Nn0])                           
                            #arezBMx[i][NumFr-NNew:NumFr]=arezBMx[i][NumFr-NNew:NumFr]-np.mean(arezBMx[i][NumFr-NNew:Nn0])+np.mean(Arr_x[icl][i][NumFr-NNew:Nn0])
                            arezBMx[i][NumFr-NNew:NumFr]=arezBMx[i][NumFr-NNew:NumFr]+Arr_x[icl][i][NumFr-NNew-1]
                            #arezBMx[i][0:NumFr-NNew]=Arr_0[icl][i][0:NumFr-NNew].copy()
                            #arezBMx[i]= savgol_filter(arezBMx[i], 11, 5) 
                            arezBMx[i][0:NumFr-NNew]=Arr_x[icl][i][0:NumFr-NNew].copy()
                           
                        for j in range(l*NChan,(l+1)*NChan):
                            for i in range(int(NumFr)):
                                ArrRezMx[icl][i][NumFrX[j]][NumFrY[j]]=max(ArrRezMx[icl][i][NumFrX[j]][NumFrY[j]],arezBMx[j-l*NChan][i])
                                ArrRezMn[icl][i][NumFrX[j]][NumFrY[j]]=min(ArrRezMn[icl][i][NumFrX[j]][NumFrY[j]],arezBMx[j-l*NChan][i])                 
                    print ("calculated %s percents"%(int((l+1)/(int(np.ceil(sz2*Ndel)))*1000)/10))
        
                ZZ=(ArrRezMx+ArrRezMn)/2    
                for icl in range(3):
                    ffZZ=[]            
                    for l in range(NumFr-NNew):
                        dd=ZZ[icl][l]-ZZ[icl][NumFr-NNew-1]
                        mnZZ=np.mean(dd)
                        ffZZ.append(np.fft.fft2(dd-mnZZ))
                    ffZZ=np.asarray(np.abs(ffZZ),float)
                    affZZ=np.max(ffZZ,0)
                    maffZZ=0.63*np.mean(affZZ)/2
                    for l in range(NumFr-NNew,NumFr):
                        dd=ZZ[icl][l]-ZZ[icl][NumFr-NNew-1]
                        mnZZ=np.mean(dd)
                        ZZ_=np.fft.fft2(dd-mnZZ)
                        aZZ_=np.abs(ZZ_)+1e-32
                        mZZ_=0.63*np.mean(aZZ_)                
                        fZZ=np.fft.ifft2((ZZ_/aZZ_)*(1*(aZZ_>mZZ_))*affZZ)#*(affZZ>maffZZ))
                        ZZ[icl][l]=fZZ.real+mnZZ+ZZ[icl][NumFr-NNew-1]
              
                if hh==0:
                    ArrRez=ZZ
                else:
                    ArrRez=(ArrRez*hh+ZZ)/(hh+1)                                 
                ArrRez_=ArrRez.copy()
                ArrRez_=np.asarray(ArrRez_-(ArrRez_-255)*(ArrRez_>255),np.uint8)
                ArrRez_=np.asarray(ArrRez_-ArrRez_*(ArrRez_<0),np.uint8)
                coefX=max(gray_sz1/sz1,gray_sz2/sz2)
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                for icl in range(3):
                    for i in range(sz1):
                        for j in range(sz2):                
                            ArrRez_[icl,:,i,j]= savgol_filter(ArrRez_[icl,:,i,j], 11, 5)
                out = cv2.VideoWriter(wwrkdir_ +nmfile,fourcc, aDur, (gray_sz2,gray_sz1))
                kk=np.zeros(3,int)
                kkk=np.zeros(3,int)
                kk[icl]=0
                kkk[icl]=0
                for kk in range(NumFr):
                    frame=frame_
                    for ii in range(astep):                
                        for icl in range(3):   
                            frame[ : , : , icl] = ndimage.zoom(ArrRez_[icl][kk], coefX)[0:gray_sz1,0:gray_sz2]                       
                        frame=cv2.medianBlur(frame,11)
                        out.write(frame)                    
                    
                    cv2.imshow('frame', frame)
                    # Press Q on keyboard to  exit 
                    if cv2.waitKey(30) & 0xFF == ord('q'): 
                        break                    
                out.release()
                cv2.destroyAllWindows()
                
                hh=hh+1
                dill.dump_session(filename+("%s_%s"%(hhh,hh-1))+".ralf")              
       
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
        while kk<NumFr_ and ret:
            for ii in range(astep):
                ret, frame = cap.read()   
                if ret:
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
