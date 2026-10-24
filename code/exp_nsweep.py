"""Effect of the ring dimension n on the cheapest parameter set (Algorithm 3), k=1."""
import csv, math
from lemos import *
rows=[]
for n in (128,256,512):
    for sec in (128,192):
        for lam in (512,2048,4096):
            b=select_params(sec,lam,kappa=128,ns=(n,),qs=(7681,12289,786433),etas=(1,2,3,4,5,6,8),ks=(1,))
            if b is None: rows.append([n,sec,lam,'','','','','','','']); continue
            rows.append([n,sec,lam,b['q'],b['l'],b['eta'],round(b['lam'],1),round(b['sec'],1),b['pk'],b['sk']])
            print(rows[-1],flush=True)
w=csv.writer(open('../data/nsweep.csv','w')); w.writerow(['n','sec_target','lam_target','q','l','eta','lam_max','msis_bits','pk_bytes','sk_bytes']); w.writerows(rows)
