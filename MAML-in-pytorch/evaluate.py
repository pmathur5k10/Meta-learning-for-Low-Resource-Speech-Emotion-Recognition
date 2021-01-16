# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import logging
import copy
import argparse

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.data_loader import fetch_dataloaders
from src.data_loader import datasets_test
import os
import utils
import pickle
import math


def evaluate(model, loss_fn, meta_classes, task_lr, task_type, metrics, params, args,
             split):
    """
    Evaluate the model on `num_steps` batches.
    
    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and 
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps
    num_eval_updates = params.num_eval_updates

    # set model to evaluation mode
    # NOTE eval() is not needed since everytime task is varying and batchnorm
    # should compute statistics within the task.
    # model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(args, meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task, args.pkl_path, params)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()
        #print ( Y_que.detach().numpy() )
        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()
            X_que, Y_que = X_que.cuda(), Y_que.cuda()

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(num_eval_updates):
            Y_sup_hat = net_clone(X_sup)
            loss = loss_fn(Y_sup_hat, Y_sup)
            optim.zero_grad()
            loss.backward()
            optim.step()
        Y_que_hat = net_clone(X_que)
        loss = loss_fn(Y_que_hat, Y_que)

        # # clear previous gradients, compute gradients of all variables wrt loss
        # def zero_grad(params):
        #     for p in params:
        #         if p.grad is not None:
        #             p.grad.zero_()

        # # NOTE In Meta-SGD paper, num_eval_updates=1 is enough
        # for _ in range(num_eval_updates):
        #     Y_sup_hat = model(X_sup)
        #     loss = loss_fn(Y_sup_hat, Y_sup)
        #     zero_grad(model.parameters())
        #     grads = torch.autograd.grad(loss, model.parameters())
        #     # step() manually
        #     adapted_state_dict = model.cloned_state_dict()
        #     adapted_params = OrderedDict()
        #     for (key, val), grad in zip(model.named_parameters(), grads):
        #         adapted_params[key] = val - task_lr * grad
        #         adapted_state_dict[key] = adapted_params[key]
        # Y_que_hat = model(X_que, adapted_state_dict)
        # loss = loss_fn(Y_que_hat, Y_que)  # NOTE !!!!!!!!

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {
            metric: metrics[metric](Y_que_hat, Y_que)
            for metric in metrics
        }
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([x[metric] for x in summ])
        for metric in summ[0]
    }
    metrics_string = " ; ".join(
        "{}: {:05.6f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    print ( metrics_string )

    return metrics_mean


if __name__ == '__main__':
    from src.model import metrics
    from src.data_loader import split_emotions
    from src.data_loader import SER
    from src.model import MetaLearner

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        default='experiments/test',
        help="Directory containing params.json")
    parser.add_argument(
        '--restore_file',
        default='best',
        help="Optional, name of the file in --model_dir containing weights to \
            reload before training")  # 'best' or 'train'

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    SEED = params.SEED
    task_lr = params.task_lr
    meta_lr = params.meta_lr

    utils.set_logger(os.path.join(args.model_dir, 'eval.log'))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)

    params.in_channels = 3
    meta_train_classes, meta_test_classes = split_emotions(
        args.data_dir, SEED)
    task_type = SER

    if params.cuda:
        model = MetaLearner(params).cuda()
    else:
        model = MetaLearner(params)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    restore_path = os.path.join(args.model_dir,
                                args.restore_file + '.pth.tar')

    logging.info("Eval metrics for dataset {}".format(','.join(datasets_test)))

    test_logs = []

    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model, meta_optimizer)

    # train_metrics = evaluate(model, loss_fn, meta_train_classes,
    #                                  task_lr, task_type, metrics, params, args,
    #                                  'train')
    test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                    task_lr, task_type, metrics, params, args,
                                    'test')
    test_logs.append( test_metrics )
    
    save_dir = "experiments/emodb_ravdess_savee_iemocap/shemo"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open( os.path.join( save_dir, "shemo_eval.pkl"), 'wb' ) as f:
        pickle.dump(test_logs, f)

    plt_x = [ x / 10.0 for x in supports ]
    plt_y = [ t['f1_score'] for t in test_logs ]

    with open( os.path.join( save_dir, "shemo_eval.txt"), 'w' ) as f:
        f.write( "\n".join( map(str, plt_y) ) )

    print ( plt_y )
    utils.make_plot( save_dir, plt_x, plt_y, 'shemo')
    