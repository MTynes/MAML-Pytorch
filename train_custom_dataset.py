import torch, os
import numpy as np
import pandas as pd
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
from torch.nn import functional as F

from meta import Meta
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    torch.manual_seed(222)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    train_image_directory = args.train_dir
    validation_image_directory = args.validation_dir
    test_image_directory = args.test_dir

    mini = MiniImagenet(train_image_directory, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=50, resize=args.imgsz)

    mini_validate = MiniImagenet(validation_image_directory, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=50, resize=args.imgsz)

    mini_test = MiniImagenet(test_image_directory, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                          k_query=args.k_qry,
                          batchsz=50, resize=args.imgsz)
    db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

    ft = args.run_fine_tuning
    if ft:
        fine_tune_image_directory = args.fine_tune_dir

        mini_fine_tune = MiniImagenet(fine_tune_image_directory, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                                      k_query=args.k_qry,
                                      batchsz=50, resize=args.imgsz)
        db_fine_tune = DataLoader(mini_fine_tune, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

    epoch_params = [args.epochs, args.fine_tuning_epochs] if ft else [args.epochs]
    train_loaders = [db, db_fine_tune] if ft else [db]
    training_modes = ['Main training cycle', 'Fine tuning cycle'] if ft else ['']

    mean_test_accs = []
    mean_metrics = []

    for i in range(len(train_loaders)):
        print('\n', training_modes[i])
        for epoch in range(epoch_params[i] // 10000):
            # fetch meta_batchsz num of episode each time

            for step, (x_spt, y_spt, x_qry, y_qry, cls) in enumerate(train_loaders[i]):
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                train_losses, train_accs = maml(x_spt, y_spt, x_qry, y_qry)
                if (step % 30) == 0:
                    print('\nEpoch ', epoch)
                    print('Training accuracies: \t', train_accs)

                if step % 500 == 0:  # evaluation
                    db_validate = DataLoader(mini_validate, 1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_test = []
                    losses_all_test = []
                    losses_q_all_test = []

                    for x_spt, y_spt, x_qry, y_qry, cls in db_validate:
                        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                        losses, losses_q, accs, preds = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                        accs_all_test.append(accs)
                        losses_all_test.append(losses)
                        losses_q_all_test.append(losses_q)

                        # [b, update_step+1]
                    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    mean_loss = (np.array(losses_all_test).mean(axis=0).astype(np.float16)).mean()
                    mean_loss_q = (np.array(losses_q_all_test).mean(axis=0).astype(np.float16)).mean()

                    print('Test accuracies: \t', accs)
                    print('Mean test acc: {}  Mean support loss:  {}   Mean query loss: {}'.format(np.mean(accs), mean_loss,
                                                                                           mean_loss_q))

                    mean_test_accs.append(np.mean(accs))

                    mm = {'train_loss': np.mean(train_losses), 'train_accuracy': np.mean(train_accs),
                          'val_loss': mean_loss, 'val_query_loss': mean_loss_q, 'val_accuracy': np.mean(accs)}
                    mean_metrics.append(mm)

    print('\nHighest mean validation accuracy: ', max(mean_test_accs))

    print('\nAssessing test set....')
    final_db = mini_test
    final_test = DataLoader(final_db, 1, shuffle=True, num_workers=1, pin_memory=True)
    test_losses, test_losses_query = [], []

    predictions_and_labels = pd.DataFrame()
    for x_spt, y_spt, x_qry, y_qry, cls in final_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        losses, losses_q, accs, preds = maml.finetunning(x_spt, y_spt, x_qry, y_qry,
                                                         return_predictions=True)
        test_losses.extend([l.item() for l in losses])
        test_losses_query.extend([l.item() for l in losses_q])
        preds['true_label'] = [cls.item() for i in range(preds.shape[0])]
        predictions_and_labels = predictions_and_labels.append(preds)


    # log the mean test accuracy data for later display
    with open(args.accuracy_log_file, 'w') as f:
        f.write("\n".join([str(s) for s in mean_test_accs]))
    pd.DataFrame(mean_metrics).to_csv('mean_metrics.csv', index=False)
    print('\nMean test accuracy: {:.2f}'.format(predictions_and_labels['correct'].mean()))
    predictions_and_labels.to_csv('test_predictions_and_labels.csv', index=False)



    truth_y = predictions_and_labels['true_label']
    pred_y = predictions_and_labels['prediction']
    auc = roc_auc_score(truth_y, pred_y)
    f1_macro = f1_score(truth_y, pred_y, average='macro')
    f1_micro = f1_score(truth_y, pred_y, average='micro')
    final_val = mean_metrics[-1]
    d = {'Final Val Support Loss': final_val['val_loss'],
         'Final Val Query Loss': final_val['val_query_loss'],
         'Final Val Accuracy': final_val['val_accuracy'],
         'Test Support Loss': np.mean(test_losses),
         'Test Query Loss': np.mean(test_losses_query),
         'Test AUC': auc,
         'Test F1 Score Macro': f1_macro,
         'Test F1 Score Micro': f1_micro
         }
    metrics_summary = pd.DataFrame(d, index=[0])
    metrics_summary.to_csv('metrics_summary.csv', index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--train_dir', type=str, help='train data directory', default='/content/miniimagenet/images')
    argparser.add_argument('--fine_tune_dir', type=str, help='fine tuning data directory',
                           default='/content/all_fine_tuning_images')
    argparser.add_argument('--validation_dir', type=str, help='validation data directory',
                           default='/content/all_validation_images')
    argparser.add_argument('--test_dir', type=str, help='test data directory', default='/content/all_test_images')
    argparser.add_argument('--run_fine_tuning', default=False, type=lambda x: (str(x).lower() == 'true'),
                           help='Boolean for adding a second dataset for further training. Set as string. Case insensitive.')

    argparser.add_argument('--epochs', type=int, help='Number of epochs', default=(200 * 10000))  ##6
    argparser.add_argument('--fine_tuning_epochs', type=int, help='Number of epochs for fine tuning cycle',
                           default=(200 * 10000))  ##6
    argparser.add_argument('--n_way', type=int, help='n way',
                           default=2)  # cannot be larger than the number of categories
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)  # 15
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)  #
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--accuracy_log_file', type=str, help='Output file for mean test accuracy',
                           default='/content/mean_test_accuracy.txt')
    args = argparser.parse_args()

    main()
