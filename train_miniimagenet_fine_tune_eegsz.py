import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
import pandas as pd
from torch.nn import functional as F

from meta import Meta


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
    test_image_directory = args.test_dir
    fine_tune_image_directory = args.fine_tune_dir

    mini = MiniImagenet(train_image_directory, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=50, resize=args.imgsz)
    mini_fine_tune = MiniImagenet(fine_tune_image_directory, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                                  k_query=args.k_qry,
                                  batchsz=50, resize=args.imgsz)
    mini_test = MiniImagenet(test_image_directory, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=50, resize=args.imgsz)
    mean_test_accs = []
    mean_metrics = []

    db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
    db_fine_tune = DataLoader(mini_fine_tune, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

    epoch_params = [args.epochs, args.fine_tuning_epochs]
    train_loaders = [db, db_fine_tune]
    training_modes = ['Main training cycle', 'Fine tuning cycle']
    for i in range(len(train_loaders)):
        print('\n', training_modes[i])
        for epoch in range(epoch_params[i] // 10000):
            # fetch meta_batchsz num of episode each time

            for step, (x_spt, y_spt, x_qry, y_qry, cls) in enumerate(train_loaders[i]):
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                train_losses, train_accs = maml(x_spt, y_spt, x_qry, y_qry)
                if (step % 30) == 0:
                    print('\nEpoch ', epoch)
                    # print('\nstep:', step, '\ttraining acc:', train_accs)
                    print('Training accuracies: \t', train_accs)

                if step % 500 == 0:  # evaluation
                    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_test = []
                    losses_all_test = []
                    losses_q_all_test = []

                    for x_spt, y_spt, x_qry, y_qry, cls in db_test:
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
                    print('Mean test acc: {}  Mean loss:  {}   Mean query loss: {}'.format(np.mean(accs), mean_loss,
                                                                                           mean_loss_q))

                    mean_test_accs.append(np.mean(accs))

                    mm = {'train_loss': np.mean(train_losses), 'train_accuracy': np.mean(train_accs),
                          'test_loss': mean_loss, 'test_accuracy': np.mean(accs)}
                    mean_metrics.append(mm)

    print('\nHighest mean test accuracy: ', max(mean_test_accs))

    # TODO set to final test set
    final_db = mini_test
    final_test = DataLoader(final_db, 1, shuffle=True, num_workers=1, pin_memory=True)

    predictions_and_labels = pd.DataFrame()
    predictions_and_labels_list = []
    predictions_and_labels_classes = []
    for x_spt, y_spt, x_qry, y_qry, cls in final_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        losses, losses_q, accs, preds = maml.finetunning(x_spt, y_spt, x_qry, y_qry,
                                                         return_predictions=True)
        preds['true_label'] = [cls.item() for i in range(preds.shape[0])]
        predictions_and_labels = predictions_and_labels.append(preds)
        predictions_and_labels_classes.append(cls)
        predictions_and_labels_list.append(preds)


    # log the mean test accuracy data for display later
    with open(args.accuracy_log_file, 'w') as f:
        f.write("\n".join([str(s) for s in mean_test_accs]))
    pd.DataFrame(mean_metrics).to_csv('mean_metrics.csv', index=False)
    predictions_and_labels.to_csv('test_predictions_and_labels.csv', index=False)
    print('shape of predictions and labels df: ')
    print(predictions_and_labels.shape)
    print('length of classes: ')
    print(len(predictions_and_labels_classes))
    print('length of predictions list')
    print(len(predictions_and_labels_list))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--train_dir', type=str, help='train data directory', default='/content/miniimagenet/images')
    argparser.add_argument('--fine_tune_dir', type=str, help='fine tuning data directory',
                           default='/content/all_fine_tuning_images')
    argparser.add_argument('--validation_dir', type=str, help='validation data directory',
                           default='/content/all_validation_images')
    argparser.add_argument('--test_dir', type=str, help='test data directory', default='/content/all_test_images')
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
