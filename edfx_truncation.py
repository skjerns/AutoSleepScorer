# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:37:22 2017

@author: Simon
"""
import sleeploader
import numpy as np
from tqdm import tqdm
#datadir = 'C:\sleep\cshs50'

datadir = 'C:\sleep\edfx'
sleep = sleeploader.SleepDataset(datadir)
sleep.load()#_object()

data = sleep.data
hypno = sleep.hypno
new_data = []
new_hypno = []
for d, h in zip(data,hypno):
    if d.shape[0]%3000!=0: d = d[:len(d)-d.shape[0]%3000]
    d = d.reshape([-1,3000,3])
    if 9 in h:
        delete = np.where(h==9)[0]
#        if len(delete)>0:print('deleting {} epochs'.format(len(delete)))
        d = np.delete(d, delete, 0)
        h = np.delete(h, delete, 0)
    begin = np.where(h-np.roll(h,1)!=0)[0][0]-300
    end = np.where(h-np.roll(h,1)!=0)[0][-1]+30
    
    d = d[begin:end]
    h = h[begin:end]

    d = d.ravel()
    new_hypno.append(h)
    new_data.append(d*1000000)
    print (len(h), len(d)//3000/3)

    

sleep.data = new_data
sleep.hypno = new_hypno
sleep.save_object()
a,b,c = sleep.get_all_data()