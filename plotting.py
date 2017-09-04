# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:10:40 2017

@author: Simon
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import re
import sdill as dill
from sklearn.metrics import f1_score
import tools
cmap = sns.cubehelix_palette(8, start=2.8, rot=-.1, as_cmap=True)
fsize = np.array((4,3.5))

#%%

pkl = pickle.load(open('.\\results\\results_recurrent_', 'rb'))
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']
for key in pkl:
    confmat = [x[4] for x in pkl[key]]
    tools.plot_confusion_matrix ('dataset_'+ key + '.png',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title=key)

pkl = pickle.load(open('.\\results\\results_recurrent_seqlen-rnn6.pkl', 'rb'))
#pkl['feat-LSTM'] = pkl.pop('pure_rnn_do_6')
confmat = [x[4] for x in pkl[list(pkl.keys())[0]]]
tools.plot_confusion_matrix ('recurrent_'+ key +'_newest.png',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='Feat-LSTM')


pkl = pickle.load(open('.\\results_recurrent_morel2.pkl', 'rb'))
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']
for key in pkl:
    confmat =  pkl[key][4]
#    plt.close('all')
    tools.plot_confusion_matrix ('transfer_'+ key + '.png',confmat, t_label,figsize=fsize, perc=True,cmap=cmap, title=key)
#%% electrodes best plot
pkl = pickle.load(open('.\\results\\results_electrodes_morel2.pkl', 'rb'))
pkl.update(pickle.load(open('.\\results\\new_results_electrodes_feat.pkl', 'rb')))
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']
confmat = [x[4] for x in pkl['annall']]
tools.plot_confusion_matrix ('confmat_ann.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, cbar=False)
confmat = [x[4] for x in pkl['cnn3morel2 all']]
tools.plot_confusion_matrix ('confmat_cnn.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap)


#%% Differenceplot

pkl = pickle.load(open('.\\results\\results_electrodes_morel2.pkl', 'rb'))
pkl.update(pickle.load(open('.\\results\\new_results_electrodes_feat.pkl', 'rb')))

t_label = ['W', 'S1', 'S2', 'SWS', 'REM']

cnn_eeg = np.mean([x[4] for x in pkl['cnn3morel2 eeg']], 0)
cnn_eog = np.mean([x[4] for x in pkl['cnn3morel2 eog']], 0)
cnn_emg = np.mean([x[4] for x in pkl['cnn3morel2 emg']], 0)
cnn_all = np.mean([x[4] for x in pkl['cnn3morel2 all']], 0)

ann_eeg = np.mean([x[4] for x in pkl['anneeg']], 0)
ann_eog = np.mean([x[4] for x in pkl['anneeg+eog']], 0)
ann_emg = np.mean([x[4] for x in pkl['anneeg+emg']], 0)
ann_all = np.mean([x[4] for x in pkl['annall']], 0)

tools.plot_difference_matrix('cnn_eeg.pdf', cnn_eeg, cnn_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG')
tools.plot_difference_matrix('cnn_eog.pdf', cnn_eog, cnn_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EOG')
tools.plot_difference_matrix('cnn_emg.pdf', cnn_emg, cnn_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EMG')

tools.plot_difference_matrix('ann_eeg.pdf', ann_eeg, ann_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG', cbar=False)
tools.plot_difference_matrix('ann_eog.pdf', ann_eog, ann_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EOG', cbar=False)
tools.plot_difference_matrix('ann_emg.pdf', ann_emg, ann_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EMG', cbar=False)


#%%
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']

pkl = pickle.load(open('.\\results\\new_results_recurrent.pkl', 'rb'))
rec_ann = np.mean([x[4] for x in pkl['pure_rnn_do']], 0)
pkl = pickle.load(open('.\\results\\results_recurrent_morel2.pkl', 'rb'))
rec_cnn = np.mean([x[4] for x in pkl['LSTM moreL2_fc1']], 0)

tools.plot_confusion_matrix ('conf_feat-lstm.pdf', rec_ann, t_label,figsize=fsize, perc=True, cmap=cmap, title='', cbar=False)
tools.plot_confusion_matrix ('conf_cnn+lstm.pdf', rec_cnn, t_label,figsize=fsize, perc=True, cmap=cmap, title='')

tools.plot_difference_matrix('diff-cnn-lstm.pdf', rec_ann, rec_cnn , t_label,figsize=fsize, perc=True, title='')

plt.close('all')

#for key in pkl:
#    confmat = pkl[key]]
#    plt.close('all')
#    
#    tools.plot_confusion_matrix ('recurrent_'+ key + '.eps',np.mean(confmat,0), target, perc=True)
#%%

pkl = pickle.load(open('.\\results\\results_recurrent_emsa', 'rb'))
confmat = [x[4] for x in pkl[list(pkl)[0]]]
tools.plot_confusion_matrix ('dataset_'+ key + '.png',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='EMSA')

t_label = ['W', 'S1', 'S2', 'SWS', 'REM']
for key in pkl:
    confmat = pkl[key][4]
    tools.plot_confusion_matrix ('dataset_'+ key + '.png',confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title=key)



#%% Datasets
pkl = pickle.load(open('.\\results\\results_recurrent_emsa', 'rb'))
confmat = [x[4] for x in pkl[list(pkl)[1]]]
tools.plot_confusion_matrix ('dataset_emsaad.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='EMSAad')

pkl = pickle.load(open('.\\results\\results_dataset_emsach.pkl', 'rb'))
confmat = [x[4] for x in pkl[list(pkl)[1]]]
tools.plot_confusion_matrix ('dataset_emsach.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='EMSAch')

pkl = pickle.load(open('.\\results\\results_recurrent_edfx', 'rb'))
confmat = [x[4] for x in pkl[list(pkl)[1]]]
tools.plot_confusion_matrix ('dataset_edfx.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='Sleep-EDFx')

pkl = pickle.load(open('.\\results\\results_recurrent_vinc', 'rb'))
confmat = [x[4] for x in pkl[list(pkl)[1]]]
tools.plot_confusion_matrix ('dataset_vinc.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='UCD')


#pkl = pickle.load(open('.\\results\\results_recurrent_cshs100', 'rb'))
#confmat = [x[4] for x in pkl[list(pkl)[0]]]
#tools.plot_confusion_matrix ('dataset_cshs100.pdf',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='CCSHS100')

#%% Transfer confmats
pkl = pickle.load(open('.\\results_transfer_cshs50_cshs50', 'rb'))
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']

key = 'cshs100'
confmat = pkl[key][4]
tools.plot_confusion_matrix ('transfer_cshs50_{}.pdf'.format(key),confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title='CCSHS100')

key = 'edfx'
confmat = pkl[key][4]
tools.plot_confusion_matrix ('transfer_cshs50_{}.pdf'.format(key),confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title='Sleep-EDFx')


key = 'emsaad'
confmat = pkl[key][4]
tools.plot_confusion_matrix ('transfer_cshs50_{}.pdf'.format(key),confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title='EMSAad')

key = 'emsach'
confmat = pkl[key][4]
tools.plot_confusion_matrix ('transfer_cshs50_{}.pdf'.format(key),confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title='EMSAch')

key = 'vinc'
confmat = pkl[key][4]
tools.plot_confusion_matrix ('transfer_cshs50_{}.pdf'.format(key),confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title='UCD')

key = 'vinc_scaled'
confmat = pkl[key][4]
tools.plot_confusion_matrix ('transfer_cshs50_{}.pdf'.format(key),confmat, t_label,figsize=fsize, perc=True, cmap=cmap, title='UCD z-scored')

#%% hypnograms

pkl = pickle.load(open('.\\results_transfer_cshs50_cshs50', 'rb'))
pred, targ, _ = pkl['cshs100'][5]
targ = np.roll(targ,4)
sub_pred = pred[5390:6560]
sub_targ = targ[5390:6560]
acc = np.mean(sub_targ==sub_pred)
f1 = f1_score(sub_targ, sub_pred, average='macro')

plt.figure(figsize=[8,3])
ax = plt.subplot(111)
tools.plot_hypnogram(sub_pred, c ='orangered', title='Ground Truth',ax1=ax)
tools.plot_hypnogram(sub_targ, c='royalblue', title='Prediction', ax1=ax, linewidth=1.9)
plt.legend(['Human Scorer','CNN+LSTM'], loc='lower right')
#plt.savefig('./plots/hypnogram_truth.pdf')
plt.savefig('./plots/hypnogram_prediction.pdf')

#%% distribution of predictions

preds = dill.load('predictions.pkl')
prob = preds['cnn_pred']
targ = preds['cnn_target']
pred = np.argmax(prob,1)

correct = np.max(prob[targ==pred],1)
wrong   =  np.max(prob[targ!=pred],1)

plt.subplot(1,2,1)
sns.distplot(correct,bins=200);
plt.title('Correct')
plt.subplot(1,2,2)
sns.distplot(wrong);
plt.title('Wrong')

#%% Plots for presentation cnn

plt.figure(figsize=[8,3])
test_acc     = np.array([0.837,	0.850,	0.842,	0.848])
test_acc_min = np.array([0.820,	0.824,	0.811,	0.818])
test_acc_max = np.array([0.850,	0.872,	0.873,	0.869])


plt.plot(test_acc, 'bo')
plt.errorbar(np.arange(4),test_acc , [test_acc - test_acc_min, test_acc_max - test_acc],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')

plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])

test_f1     = np.array([0.710,	0.734,	0.732,	0.743])
test_f1_min = np.array([0.691,	0.706,	0.710,	0.721])
test_f1_max = np.array([0.736,	0.764,	0.761,	0.760])

plt.figure(figsize=[8,3])
plt.plot(test_f1, 'go')
plt.errorbar(np.arange(4),test_f1 , [test_f1 - test_f1_min, test_f1_max - test_f1],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1-score')

plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])

#%% Plots for presentation rnn
plt.figure(figsize=[5,3])
rec_acc     = np.array([0.848, 0.856])
rec_acc_min = np.array([0.818, 0.836])
rec_acc_max = np.array([0.869, 0.876])



plt.plot(rec_acc[0], 'bo')
plt.errorbar([0] ,rec_acc[:1] , [rec_acc[:1] - rec_acc_min[:1], rec_acc_max[:1] - rec_acc[:1]],
             fmt='.k', ecolor='gray', lw=1)
plt.plot(1,rec_acc[1], 'go')
plt.errorbar([1] ,rec_acc[1:] , [rec_acc[1:] - rec_acc_min[1:], rec_acc_max[1:] - rec_acc[1:]],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')
plt.xticks(np.arange(2), ['CNN','CNN+LSTM'])
plt.xlim([-1,2])

plt.figure(figsize=[5,3])
rec_f1     = np.array([0.743,0.764])
rec_f1_min = np.array([0.721,0.740])
rec_f1_max = np.array([0.760,0.785])

plt.plot(rec_f1[0], 'bo')
plt.errorbar(0,rec_f1[:1] , [rec_f1[:1] - rec_f1_min[:1], rec_f1_max[:1] - rec_f1[:1]],
             fmt='.k', ecolor='gray', lw=1)
plt.plot(1,rec_f1[1], 'go')
plt.errorbar(1,rec_f1[1:] , [rec_f1[1:] - rec_f1_min[1:], rec_f1_max[1:] - rec_f1[1:]],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1-score')
plt.xticks(np.arange(2), ['CNN','CNN+LSTM'])
plt.xlim([-1,2])

#%% plotting hand vs automatic
feat_acc     = np.array([0.812,  	0.834,  	0.836,  	0.847])
feat_acc_min = np.array([0.772,	0.805,	0.806,	0.823])
feat_acc_max = np.array([0.832,  	0.853,  	0.857,  	0.863  ])
feat_f1      = np.array([0.647,  	0.677,  	0.714 , 	0.730 ])
feat_f1_min  = np.array([0.617,	0.648,	0.687,	0.706])
feat_f1_max  = np.array([0.667,  	0.695,  	0.739,  	0.754])
rec_acc     = np.array([0.856])
rec_acc_min = np.array([0.836])
rec_acc_max = np.array([0.876])
rec_f1     = np.array([0.764])
rec_f1_min = np.array([0.740])
rec_f1_max = np.array([0.785])
frec_acc     = np.array([0.853])
frec_acc_min = np.array([0.815])
frec_acc_max = np.array([0.883])
frec_f1      = np.array([0.754])
frec_f1_min  = np.array([0.713])
frec_f1_max  = np.array([0.783])

plt.figure(figsize=[6,3])
plt.plot(feat_acc, 'go')
plt.errorbar(np.arange(4),feat_acc , [feat_acc - feat_acc_min, feat_acc_max - feat_acc], fmt='.k', ecolor='gray', lw=1)
plt.plot(np.arange(4)+0.2, test_acc, 'bo')
plt.errorbar(np.arange(4)+0.2, test_acc , [test_acc - test_acc_min, test_acc_max - test_acc], fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')
plt.legend(['Handcrafted with ANN','Automatic with CNN']      ,loc=4 )
plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])
plt.figure(figsize=[6,3])
plt.plot(feat_f1, 'go')
plt.errorbar(np.arange(4),feat_f1 , [feat_f1 - feat_f1_min, feat_f1_max - feat_f1], fmt='.k', ecolor='gray', lw=1)
plt.plot(np.arange(4)+0.2, test_f1, 'bo')
plt.errorbar(np.arange(4)+0.2, test_f1 , [test_f1 - test_f1_min, test_f1_max - test_f1], fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1')
plt.legend(['Handcrafted with ANN','Automatic with CNN']      ,loc=4 )
plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])

plt.figure(figsize=[5,3])
plt.plot(frec_acc, 'go')
plt.errorbar(np.arange(1),frec_acc , [frec_acc - frec_acc_min, frec_acc_max - frec_acc],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')
plt.xlim([-1,2])
plt.plot(0.2,rec_acc, 'bo')
plt.errorbar(np.arange(1)+0.2,rec_acc , [rec_acc - rec_acc_min, rec_acc_max - rec_acc],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy Temporal')
plt.legend(['Handcrafted with RNN','Automatic with CNN+LSTM']      ,loc=4 )
plt.xticks(np.arange(1), ['All'])
plt.xlim([-1,2])
plt.figure(figsize=[5,3])
plt.plot(frec_f1, 'go')
plt.errorbar(np.arange(1),frec_f1 , [frec_f1 - frec_f1_min, frec_f1_max - frec_f1],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy Temporal')
plt.xlim([-1,2])
plt.plot(0.2,rec_f1, 'bo')
plt.errorbar(np.arange(1)+0.2,rec_f1 , [rec_f1 - rec_f1_min, rec_f1_max - rec_f1],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1 Temporal')
plt.legend(['Handcrafted with RNN','Automatic with CNN+LSTM']      ,loc=4 )
plt.xticks(np.arange(1), ['All'])
plt.xlim([-1,2])