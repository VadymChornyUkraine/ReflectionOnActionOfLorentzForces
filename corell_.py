import urllib.request, json
import pandas as pd
import dateutil.parser
import numpy as np
from operator import itemgetter
import time as tm
import csv
import gtcsv as gtscvv

wrkdir = r"/home/vacho/Документи/Work/W14_7/WWW/"

Lo=1
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

api_key = 'ONKTYPV6TAMZK464' 
interv="Daily"
aDecm=2
    
def decimat(adat_):
    if Lo:
        adat_=np.log(adat_)
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
        dat=dat[len(dat)-aLengt:]
        dat=dat[:len(dat)-2*aDecm]
        dat=np.asarray(dat,float)
        arrr=np.asarray(dat,float)
        
    return arrr,adat_

gtscvv.getcsv(WhO,1,wrkdir)
arrrxxR=[]
nams=[]
lenar=[]
Lengt=150
for i in range(len(WhO)):
    i0=0
    while i0<6 and not i0<0:
        try:            
            arrrxx1,adat1_=loaddata(Lengt,WhO[i],0)                        
            arrrxxR.append(arrrxx1)
            lenar.append(len(arrrxx1))
            nams.append(WhO[i])
            i0=-1
        except:
            i0=i0+1
            #gtscvv.getcsv(WhO,1,wrkdir)
            #tm.sleep(10)
           
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
        arrrxxR_.append((np.log(aaer[ii])))
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
sns.set(font_scale=0.66)
sns.heatmap(correlation_mat, 
        xticklabels=nnams_,
        yticklabels=nnams_)

plt.show()
        