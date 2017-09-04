# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:15:31 2017

@author: Simon
"""



import os
male = 0
female = 0
trainset= 'cssdf'
normmode='asd'
edfs =[s for s in os.listdir('D:/sleep/edfx') if s.endswith('.edf')]
for edf in edfs:
    with open('D:/sleep/edfx/' + edf, 'rb') as f:
        a = str(f.readline()[:30])
        if 'Male' in a:
            male+=1
        elif 'Female' in a:
            female+=1
        else :
            print(a)
print()