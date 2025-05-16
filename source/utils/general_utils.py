import time
import json
from torch import nn
import os 
import random
import numpy as np 
import torch
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def update_df_metrics(df_metrics, epoch, metrics_dict, save_df_path, plot_df_path, comment):

    if epoch == 0:
        os.makedirs(save_df_path, exist_ok=True)
        os.makedirs(plot_df_path, exist_ok=True)
        df_metrics = {'epoch':[]}
        for metric_name, _ in metrics_dict.items():
            df_metrics[metric_name] = []

    df_metrics['epoch'].append(epoch)
    for metric_name, metric in metrics_dict.items():
        df_metrics[metric_name].append(metric)

    df=pd.DataFrame.from_dict(df_metrics,orient='index').transpose()
    df.to_csv(save_df_path / f'{comment}_df_metrics.csv')

    for col in df.columns:
        if col != 'epoch':
            plt.plot(df[col])
            plt.title(col)
            plt.savefig(plot_df_path / f'plot_{col}.png')
            plt.close()
    return df_metrics

class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())

        print(f'values: {values}')

        if len(values)>1:
            for value in values:
                # split it into key and value
                key, value = value.split('=')
                # assign into dictionary
                getattr(namespace, self.dest)[key] = value

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return
    
def set_reproducibility(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    #torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    return

def monitor_training(last_logging, avg_losses_epoch, loss, rescaled_loss, args, lr, start_time, epoch, step, stats_file):
    avg_losses_epoch['avg_tot_loss_epoch'+ f'_{str(epoch)}'] = loss.item()
    avg_losses_epoch['avg_tot_loss_rescaled_epoch'+ f'_{str(epoch)}'] = rescaled_loss.item()
    
    current_time = time.time()
    
    stats = dict(
    epoch=epoch,
    step=step,
    rescaled_loss = rescaled_loss.item(),
    loss=loss.item(),
    time=int(current_time - start_time),
    day_time = str(datetime.now()),
    lr=lr,
    )
    
    if step == 1: #(current_time - last_logging > args.log_freq_time) or (step == 0):
        print(json.dumps(stats), file=stats_file)
        last_logging = current_time

    stats_names_list = ['avg_tot_loss_rescaled','avg_tot_loss']

    return stats_names_list, avg_losses_epoch, last_logging

def monitor_training_vicreg(last_logging, avg_losses_epoch, loss, sim_loss, std_loss, cov_loss, args, lr, start_time, epoch, step, stats_file):
    avg_losses_epoch['avg_tot_loss_epoch'+ f'_{str(epoch)}'] += loss.item()
    avg_losses_epoch['avg_sim_loss_epoch'+ f'_{str(epoch)}'] += sim_loss.item()
    avg_losses_epoch['avg_w_sim_loss_epoch'+ f'_{str(epoch)}'] += sim_loss.item() * args.sim_coeff
    avg_losses_epoch['avg_std_loss_epoch'+ f'_{str(epoch)}']+= std_loss.item()
    avg_losses_epoch['avg_w_std_loss_epoch'+ f'_{str(epoch)}']+= std_loss.item() * args.std_coeff
    avg_losses_epoch['avg_cov_loss_epoch'+ f'_{str(epoch)}']+= cov_loss.item()
    avg_losses_epoch['avg_w_cov_loss_epoch'+ f'_{str(epoch)}']+= cov_loss.item()* args.cov_coeff
    stats_names_list = ['avg_tot_loss','avg_sim_loss', 'avg_w_sim_loss', 'avg_std_loss', 'avg_w_std_loss', 'avg_cov_loss', 'avg_w_cov_loss']
    
    current_time = time.time()
    stats = dict(
    epoch=epoch,
    step=step,
    loss=loss.item(),
    sim_loss = sim_loss.item(),
    w_sim_loss =  sim_loss.item() * args.sim_coeff,
    std_loss = std_loss.item(),
    w_std_loss = std_loss.item() * args.std_coeff,
    cov_loss = cov_loss.item(),
    w_cov_loss = cov_loss.item()* args.cov_coeff,
    time=int(current_time - start_time),
    lr=lr,
    )         
    if step == 1: #(current_time - last_logging > args.log_freq_time) or
        print(json.dumps(stats), file=stats_file)
        last_logging = current_time

    return stats_names_list, avg_losses_epoch, last_logging

def compute_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Convert total size to MB for a more human-readable format
    total_size_MB = total_size / (1024**2)

    return num_params, total_size_MB