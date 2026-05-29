import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    jaccard_score, balanced_accuracy_score, matthews_corrcoef, f1_score, confusion_matrix


def evaluate(model, data_loader, isoform, prj_dir, thre_binary=0.0, csv_save=False):
    model.eval()
    label_predict = torch.tensor([], dtype=torch.float32).cuda()
    label_true = torch.tensor([], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    smiles_list = []
    epoch_loss = 0.0
    num_samples = 0 
    with torch.no_grad():
        for data_batch in data_loader:
            label_predict_batch = model(data_batch)
            label_true_batch = data_batch['target']['label'].to(torch.long)
            smiles_list.extend(data_batch['input']['smiles'])
            loss = criterion(label_predict_batch, label_true_batch)
            epoch_loss += loss.item()
            num_samples += len(label_predict_batch)
            label_predict = torch.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = torch.cat((label_true, label_true_batch.detach()), dim=0)
    epoch_loss_mean = epoch_loss / num_samples
    label_predict = torch.softmax(label_predict, dim=1)
    label_predict = label_predict.cpu().numpy()
    predict_label_bin = []
    label_true = label_true.cpu().numpy()
    y_pred = label_predict[:, 1]
    if thre_binary:
        predict_label_bin = []
        for prob in y_pred:
            if prob>=thre_binary:
                predict_label_bin.append(1)
            else:
                predict_label_bin.append(0)
        predict_label_bin = np.array(predict_label_bin)    
    else:
        thre_acc = []
        for thre in np.arange(0, 1.01, 0.01):
            preds_bi = [1 if x >=thre else 0 for x in y_pred]
            accuracy = round(accuracy_score(label_true, preds_bi), 3)
            thre_acc.append([thre, accuracy])
        thre_acc = np.array(thre_acc)
        maxthre_idx = np.argmax(thre_acc, axis=0)[1]
        maxthre = thre_acc[maxthre_idx, 0]
        # maxmacc = thre_acc[maxthre_idx, 1]
        predict_label_bin = [1 if x >=maxthre else 0 for x in y_pred] 
           
    if csv_save:
        df = pd.DataFrame({'smiles':smiles_list, 'label_true': label_true, 'predict_label_bin': predict_label_bin, 'predict_label_prob': label_predict[:, 1]})
        df.to_csv(f"{prj_dir}/datasets/{isoform}/result/label_predict_test_{isoform}.csv", index=False)

    # tn, fp, fn, tp = confusion_matrix(label_true, predict_label_bin).ravel()
    auc_roc = round(roc_auc_score(label_true, label_predict[:, 1]), 3)
    auc_prc = round(average_precision_score(label_true, label_predict[:, 1]), 3)
    accuracy = round(accuracy_score(label_true, predict_label_bin), 3)
    precision = round(precision_score(label_true, predict_label_bin), 3)
    recall = round(recall_score(label_true, predict_label_bin), 3)
    f1score = round(f1_score(label_true, predict_label_bin), 3)
    mcc = round(matthews_corrcoef(label_true, predict_label_bin), 3)
    jaccard = round(jaccard_score(label_true, predict_label_bin), 3)
    balanced_accuracy = round(balanced_accuracy_score(label_true, predict_label_bin), 3)
    if thre_binary:
        metric = {'auc_roc': auc_roc, 'auc_prc': auc_prc, 'accuracy': accuracy, "balanced_accuracy": balanced_accuracy,
                'precision': precision, 'recall': recall, 'f1_score': f1score, 'mcc':mcc, "jaccard": jaccard}
    else:
        metric = {'auc_roc': auc_roc, 'auc_prc': auc_prc, 'accuracy': accuracy, "balanced_accuracy": balanced_accuracy,
                'precision': precision, 'recall': recall, 'f1_score': f1score, 'mcc':mcc, "jaccard": jaccard,
                'decision_threshold': maxthre}        
    return metric, epoch_loss_mean

def inference(model, data_loader, isoform, prj_dir, thre_binary=0.5):
    model.eval()
    label_predict = torch.tensor([], dtype=torch.float32).cuda()
    smiles_list = []
    num_samples = 0 
    with torch.no_grad():
        for data_batch in data_loader:
            label_predict_batch = model(data_batch)
            smiles_list.extend(data_batch['input']['smiles'])
            num_samples += len(label_predict_batch)
            label_predict = torch.cat((label_predict, label_predict_batch.detach()), dim=0)
    label_predict = torch.softmax(label_predict, dim=1)
    label_predict = label_predict.cpu().numpy()
    predict_label_bin = []
    y_pred = label_predict[:, 1]
    predict_label_bin = []
    for prob in y_pred:
        if prob>=thre_binary:
            predict_label_bin.append(1)
        else:
            predict_label_bin.append(0)
    predict_label_bin = np.array(predict_label_bin)    
    df = pd.DataFrame({'smiles':smiles_list, 'predict_label_bin': predict_label_bin, 'predict_label_prob': label_predict[:, 1]})
    df.to_csv(f"{prj_dir}/datasets/{isoform}/result/label_predict_test_{isoform}.csv", index=False)