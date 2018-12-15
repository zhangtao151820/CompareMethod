# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:59:56 2018

@author: zhang
"""
import keras
import matplotlib.pyplot as plt
import time
import numpy as np

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        self.mytime = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        self.mytime['batch'].append(time.time())

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.mytime['epoch'].append(time.time());

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(figsize=(20,15))
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type,fontsize=25);  plt.xticks(fontsize=20);   
        plt.ylabel('acc-loss',fontsize=25); 
        plt.ylim(0, 1.2); plt.yticks(fontsize=20);
        plt.legend(loc="upper right",ncol = 2,fontsize = 24);
        plt.show()
        
        
    def loss_plotMytime(self, loss_type):
        #iters = self.mytime[loss_type] - self.mytime[loss_type][0];
        iters = np.array(self.mytime[loss_type]);
        myiters = iters -iters[0];
        allepochs = range(len(self.losses[loss_type]))
        fig = plt.figure(figsize=(20,15))
        ax1 = fig.add_subplot(111)
        # acc
        ax1.plot(allepochs, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        ax1.plot(allepochs, self.losses[loss_type], 'g', label='train loss')
        ax1.lines.pop(0);   ax1.lines.pop(0);
        
        ax2 = plt.twiny();   # 顶上X轴为时间信息
        # acc
        ax2.plot(myiters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        ax2.plot(myiters, self.losses[loss_type], 'g', label='train loss')
        
        if loss_type == 'epoch':
            # val_acc
            ax1.plot(allepochs, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            ax1.plot(allepochs, self.val_loss[loss_type], 'k', label='val loss')
            ax1.lines.pop(0);  ax1.lines.pop(0);
            
            # val_acc
            ax2.plot(myiters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            ax2.plot(myiters, self.val_loss[loss_type], 'k', label='val loss')
             

        ax1.set_xlabel(loss_type,fontsize=25);     ax2.set_xlabel('time(s)',fontsize=25);    
        ax2.legend(loc="upper right",ncol = 2,fontsize = 24); ax1.set_ylabel('acc-loss',fontsize=25);       
        ax1.locator_params("x", nbins = 20);  ax2.locator_params("x", nbins = 20);
          
        ax1.grid(True);
        
        ax = plt.gca();     ax.set_ylim(0, 1.2);   
        plt.setp(ax1.get_xticklabels(), fontsize=20);  plt.setp(ax2.get_xticklabels(), fontsize=20);
        plt.setp(ax1.get_yticklabels(), fontsize=20);
        plt.show();
        