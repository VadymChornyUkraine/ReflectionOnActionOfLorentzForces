import urllib.request
import time as tm

def getcsv(WhO,xYears,wrkdir):
    tm0=int(tm.time())
    tm0=tm0-24*60*60
    tm1=tm0-xYears*365*24*60*60
    
    for i in range(len(WhO)):
        nm=WhO[i]
        anm="https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&includeAdjustedClose=true"%(nm,tm1,tm0)   
        try:
            webUrl  = urllib.request.urlopen(anm,timeout=2)
            data = webUrl.read().decode()
            with open( wrkdir+nm+'.csv', 'w' ) as output:
                output.write( data )
        except:
            anm=anm
            
            
wrkdir = r"/home/vacho/Документи/Work/W14_7/WX4/"

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

getcsv(WhO,2,wrkdir)