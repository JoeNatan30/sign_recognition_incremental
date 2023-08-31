
import logging
import torch
import csv
import wandb

import torch.nn.functional as F

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    data_length = len(dataloader)

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs).expand(1, -1, -1)

        loss = criterion(outputs[0], labels[0])
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        #scheduler.step(running_loss.item() / len(dataloader))
        scheduler.step()

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all)


def evaluate(model, dataloader, cel_criterion, device, print_stats=False):

    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(300)}

    data_length = len(dataloader)

    k = 5 # top 5 (acc)

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        #print(f"iteration {i} in evaluate")
        outputs = model(inputs).expand(1, -1, -1)

        loss = cel_criterion(outputs[0], labels[0])
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0].item())][0] += 1
            pred_correct += 1
        
        if int(labels[0][0]) in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
            pred_top_5 += 1

        stats[int(labels[0][0].item())][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all), stats


###########################################################################
#
# DISTILLATION TRAIN
#
######################################
def cross_distillation_loss(outputs, old_outputs, T):
    # Compute the distilling loss on old classes based on the modified cross-distillation loss equation

    p_old = F.softmax(old_outputs / T, dim=1)
    p_new = F.softmax(outputs / T, dim=1)

    loss_distillation = torch.mean(-torch.sum(p_old * torch.log(p_new[:,:p_old.shape[1]]), dim=1))
    return loss_distillation

def train_distillation_epoch(model_teacher, model_student, dataloader, criterion, optimizer, alpha, T, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    data_length = len(dataloader)

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs_teacher = model_teacher(inputs).expand(1, -1, -1)
        outputs_student = model_student(inputs).expand(1, -1, -1)

        distillation_loss = cross_distillation_loss(outputs_student[0], outputs_teacher[0], T)
        crossEntropy_loss = criterion(outputs_student[0], labels[0])

        loss = alpha * distillation_loss + (1 - alpha) * crossEntropy_loss

        loss.backward()

        optimizer.step()
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs_student, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        #scheduler.step(running_loss.item() / len(dataloader))
        scheduler.step()

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all)

###########################################################################
#
# DISTILLATION EVALUATIONS
#
######################################

def evaluate_distillation(model_teacher, model_student, dataloader, cel_criterion, alpha, T, device, print_stats=False):

    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    data_length = len(dataloader)

    k = 5 # top 5 (acc)

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        #print(f"iteration {i} in evaluate")
        outputs_teacher = model_teacher(inputs).expand(1, -1, -1)
        outputs_student = model_student(inputs).expand(1, -1, -1)

        distillation_loss = cross_distillation_loss(outputs_student[0], outputs_teacher[0], T)

        crossEntropy_loss = cel_criterion(outputs_student[0], labels[0])

        loss = alpha * distillation_loss + (1 - alpha) * crossEntropy_loss

        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs_student, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1
        
        if int(labels[0][0]) in torch.topk(torch.reshape(outputs_student, (-1,)), k).indices.tolist():
            pred_top_5 += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all), stats



def evaluate_top_k(model, dataloader, device, k=5):

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist():
            pred_correct += 1

        pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)

import pandas as pd

def evaluate_with_features(model, dataloader, cel_criterion, device, print_stats=False, save_results=False):

    pred_correct, pred_top_5, pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    data_length = len(dataloader)

    k = 5 # top 5 (acc)

    # create a list to store the results
    results = []

    for i, data in enumerate(dataloader):
        inputs, labels, video_name, false_seq,percentage_group,max_consec = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        #print(f"iteration {i} in evaluate, video name {video_name}, max_consec {max_consec[i]}")
        outputs = model(inputs).expand(1, -1, -1)

        loss = cel_criterion(outputs[0], labels[0])
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1
        
        if int(labels[0][0]) in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
            pred_top_5 += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

        # calculate the accuracy per instance
        acc = 1 if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]) else 0

        # append the results to the list
        results.append({
            'video_name': video_name,
            'in_range_sequences': false_seq[i].numpy()[0],
            'percentage_group': percentage_group[i].numpy()[0],
            'max_percentage': max_consec[i].numpy()[0],
            'accuracy': acc
        })

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    # convert the list to a DataFrame
    df_results = pd.DataFrame(results)

    # # save the DataFrame to a CSV file if save_results is True
    # if save_results:
    #     save_path = 'results.csv'
    #     df_results.to_csv(save_path, index=False)

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all), df_results


def generate_csv_result(run, model, dataloader, folder_path, meaning, device):

    model.train(False)
    
    submission = dict()
    trueLabels = dict()
    meaningLabels = dict()

    for i, data in enumerate(dataloader):
        inputs, labels, video_name = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs).expand(1, -1, -1)

        pred = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))
        trueLab = int(labels[0][0])

        submission[video_name] = pred
        trueLabels[video_name] = trueLab
        meaningLabels[video_name] = meaning[trueLab]

    diccionarios = [submission, trueLabels, meaningLabels]

    # Define the row names
    headers = ['videoName', 'prediction', 'trueLabel', 'class']

    full_path = folder_path+'/submission.csv'

    # create the csv and define the headers
    with open(full_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # write the acummulated data
        for key in diccionarios[0].keys():
            row = [key[0]]
            for d in diccionarios:
                row.append(d[key])
            writer.writerow(row)
    
    #artifact = wandb.Artifact(f'predicciones_{run.id}.csv', type='dataset')
    #artifact.add_file(full_path)
    #run.log_artifact(artifact)
    wandb.save(full_path)


def generate_csv_accuracy(df_stats, folder_path):

    full_path = folder_path+'/accuracy.csv'
    df_stats.to_csv(full_path, index=False, encoding='utf-8')
    wandb.save(full_path)