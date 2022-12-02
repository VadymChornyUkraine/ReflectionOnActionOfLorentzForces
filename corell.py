import urllib.request, json
import pandas as pd
import dateutil.parser
import numpy as np
from operator import itemgetter
import time as tm

Lo=1
WhO=[
"ADAUSD",
"ANKUSD",
"AVEUSD",
"BALUSD",
"BANUSD",
"BCHUSD",
"BNTUSD",
"BTCUSD",
"CLDUSD",
"CMPUSD",
"CRVUSD",
"DOGUSD",
"EOSUSD",
"ETCUSD",
"ETHUSD",
"GRTUSD",
"KNCUSD",
"LNKUSD",
"LRCUSD",
"LTCUSD",
"MATUSD",
"MKRUSD",
"MNAUSD",
"NMRUSD",
"NUCUSD",
"OMGUSD",
"SHIUSD",
"SKLUSD",
"SNXUSD",
"STOUSD",
"UMAUSD",
"UNIUSD",
"XLMUSD",
"XTZUSD",
"ZRXUSD",
"АТОUSD"
]

api_key = 'ONKTYPV6TAMZK464' 
interv="Daily"
aDecm=4
    
def decimat(adat_):
    if Lo:
        adat_=np.log(adat_)
    adatx=0
    k=0
    adat__=np.zeros(int(len(adat_)/aDecm),float)
    for i in range(int(len(adat_)/aDecm)):
        adat__[k]=np.median(adat_[i*aDecm:i*aDecm+aDecm])
        k=k+1
    if Lo:
        return np.exp(adat__[1:len(adat__)])
    else:
        return (adat__[1:len(adat__)])
    
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
        file=open(ticker1+ '.txt','r')
        arrr=np.asarray(file.readlines(),float).copy()
        if len(arrr)>aLengt:
            arrr=arrr[len(arrr)-aLengt:]
        file.close()
        
    return arrr,adat_

arrrxxR=[]
nams=[]
lenar=[]
Lengt=400
for i in range(len(WhO)):
    i0=0
    while i0<6 and not i0<0:
        try:
            arrrxx1,adat1_=loaddata(Lengt,WhO[i],1)
            arrrxxR.append(arrrxx1)
            lenar.append(len(arrrxx1))
            nams.append(WhO[i])
            i0=-1
        except:
            i0=i0+1
            tm.sleep(10)
           
llar=int(np.median(np.asarray(lenar,int)))
nnams_=[]
aaer=[]
for i in range(len(nams)):    
    if len(arrrxxR[i])>=llar:
        aer=decimat(arrrxxR[i])
        aaer.append(aer[len(aer)-int(llar/aDecm)+1:].copy())
aaer=np.asarray(aaer,float)

arrrxxR_=[]
ii=0
for i in range(len(nams)):    
    if len(arrrxxR[i])>=llar:        
        arrrxxR_.append(np.diff(np.log(aaer[ii])))
        nnams_.append(nams[ii])
        ii=ii+1
arrrxxR_=np.asarray(arrrxxR_,float)
       
srarr=np.median((arrrxxR_),axis=0)
arrrxxRR_=np.asarray(arrrxxR_,float)*0
for i in range(len(nnams_)):
    arrrxxRR_[i]=(arrrxxR_[i]-srarr)
#arrrxxRR_=arrrxxRR_.transpose()

import seaborn as sns
import matplotlib.pyplot as plt
correlation_mat = np.corrcoef(arrrxxRR_)
sns.heatmap(correlation_mat, 
        xticklabels=nnams_,
        yticklabels=nnams_)
plt.show()
        