import json, csv, math
import numpy as np
from lemos import *
# --- Table: parameter selection
rows=[]
for sec in (128,192,256):
    for lam in (0,512,1024,2048,4096):
        p=select_params(sec,lam)
        rows.append(dict(sec_target=sec,lam_target=('none' if lam is None else lam),**p))
import pandas as pd
df=pd.DataFrame(rows); df.to_csv('../data/select_table.csv',index=False)
print(df[['sec_target','lam_target','q','k','l','eta','lam','sec','pk','sk','total','rate']].round(1).to_string())
# --- sk size vs lambda
out=[]
for sec in (128,192,256):
    for lam in range(0,6001,250):
        p=select_params(sec,lam)
        out.append(dict(sec_target=sec,lam_target=lam,sk=p['sk'],pk=p['pk'],total=p['total'],l=p['l'],eta=p['eta'],q=p['q'],lam=p['lam']))
pd.DataFrame(out).to_csv('../data/sk_vs_lambda.csv',index=False)
# --- heatmap: lambda_max over (l, eta), n=256,k=1, q=7681 ; and leakage rate
etas=[1,2,3,4,5,6,8,10]; ls=list(range(2,25))
H=np.array([[lambda0(256,1,l,7681,e)-128 for l in ls] for e in etas])
R=np.array([[ (lambda0(256,1,l,7681,e))/log2_secret_space(256,l,e) for l in ls] for e in etas])
np.savetxt('../data/heat_lambda.csv',H,delimiter=',',fmt='%.1f'); np.savetxt('../data/heat_rate.csv',R,delimiter=',',fmt='%.4f')
json.dump(dict(etas=etas,ls=ls),open('../data/heat_axes.json','w'))
# --- MSIS security along l for q=7681, eta=3 and eta=1
sec=[]
for eta in (1,3):
    for l in range(2,26):
        b,cl,qu=msis_core_svp(256,1,l,7681,beta_of(256,l,eta))
        sec.append(dict(eta=eta,l=l,b=b,classical=cl,quantum=qu,lam=lambda0(256,1,l,7681,eta)-128))
pd.DataFrame(sec).to_csv('../data/msis_vs_l.csv',index=False)
print(pd.DataFrame(sec).round(1).head(30).to_string())
