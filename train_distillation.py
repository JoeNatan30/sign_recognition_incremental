# Default
import os
import random

# Three-party
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import wandb
import numpy as np
import pandas as pd

# local
from datasets.Lsp_dataset import LSP_Dataset
from spoter.gaussian_noise import GaussianNoise
from parser_default import get_default_args
from spoter.utils import train_distillation_epoch, evaluate_distillation, train_epoch, evaluate
from spoter.utils import generate_csv_result, generate_csv_accuracy
from incremental_model import incremental_model_type

parser = get_default_args()
args = parser.parse_args()

# Set device to CUDA only if applicable
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")

# Initialize all the random seeds
random.seed(args.seed)
np.random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
g = torch.Generator()
g.manual_seed(args.seed)

version = 3
args.model_type = "distillation_old"
dataset = "DGI305-AEC"
limit_type = "fixed_with_old" # fixed
previous_model_type = "distillation" #distillation base
maximun_train = 12
maximun_val = 3

dataset_reference = 60
args.prev_num_classes = 40
args.new_num_classes = 60


df_words = pd.read_csv(f"./incrementalList_V{version}.csv",encoding='utf-8', header=None)
words = list(df_words[0])

all_words = words[:args.new_num_classes]
new_words = words[args.prev_num_classes:args.new_num_classes]
old_words = words[:args.prev_num_classes]


max_patience = 200
alpha = args.prev_num_classes / args.new_num_classes # old / total
T = 2

if previous_model_type == "base":
    args.load_model_from = f"{previous_model_type}_spoter_{args.prev_num_classes}_{args.prev_num_classes}_V{version}/"
else:
    args.load_model_from = f"{previous_model_type}_spoter_{args.new_num_classes - args.prev_num_classes}_{args.prev_num_classes}_V{version}/"

args.training_set_path = f'../ConnectingPoints/split/{dataset}--{dataset_reference}--incremental--mediapipe--V{version}-Train.hdf5'
args.validation_set_path = f'../ConnectingPoints/split/{dataset}--{dataset_reference}--incremental--mediapipe--V{version}-Val.hdf5'

args.epochs = 1000
args.lr = 0.00005

PROJECT_WANDB = "SIMBig_incremental_learning"
ENTITY = "joenatan30" 
TAG = ["No_freezing",f'prev_{args.prev_num_classes}',f'new_{args.new_num_classes}', args.model_type, f'V{version}']
args.experiment_name = f'{args.model_type}_spoter_{args.prev_num_classes}_{args.new_num_classes}_V{version}'

run = wandb.init(project=PROJECT_WANDB, 
                 entity=ENTITY,
                 config=args, 
                 name=args.experiment_name, 
                 job_type="model-training",
                 tags=TAG)

config = wandb.config
wandb.watch_called = False

model_teacher, model_student = incremental_model_type(args)

# DATA LOADER
# Training set
transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
if args.prev_num_classes == args.new_num_classes:
    train_set = LSP_Dataset(args.training_set_path, words=old_words, transform=transform, have_aumentation=True, keypoints_model='mediapipe',
                        limit_type=limit_type, maximun=maximun_train)
elif limit_type == "fixed_with_old":
    train_set = LSP_Dataset(args.training_set_path, words=all_words, transform=transform, have_aumentation=True, keypoints_model='mediapipe',
                        limit_type=limit_type, maximun=maximun_train)
else:
    train_set = LSP_Dataset(args.training_set_path, words=new_words, transform=transform, have_aumentation=True, keypoints_model='mediapipe',
                        limit_type=limit_type, maximun=maximun_train)
# Validation set
val_set = LSP_Dataset(args.validation_set_path, words=all_words, keypoints_model='mediapipe', have_aumentation=False,
                      limit_type="fixed", maximun=maximun_val)
val_loader = DataLoader(val_set, shuffle=True, generator=g)

# Testing set
if args.testing_set_path:
    eval_set = LSP_Dataset(args.testing_set_path, words=words, keypoints_model='mediapipe')
    eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

else:
    eval_loader = None

train_loader = DataLoader(train_set, shuffle=True, generator=g)

#class_weight = torch.FloatTensor([1/train_set.label_freq[i] for i in range(args.new_num_classes)]).to(device)
cel_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)#, weight=class_weight)
sgd_optimizer = optim.SGD(model_student.parameters(), lr=args.lr)

epoch_start = 0

# Ensure that the path for checkpointing and for images both exist
Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
Path("out-img/").mkdir(parents=True, exist_ok=True)


print("#"*50)
print("#"*30)
print("#"*10)
print("Num Trainable Params: ", sum(p.numel() for p in model_student.parameters() if p.requires_grad))
print("#"*10)
print("#"*30)
print("#"*50)


# MARK: TRAINING
train_acc, val_acc = 0, 0
losses, train_accs, val_accs, val_accs_top5 = [], [], [], []
lr_progress = []
top_train_acc, top_val_acc = 0, 0
checkpoint_index = 0

model_student.train(True)
model_student.to(device)
model_teacher.to(device)

patience = 0

#################################################################
#
# FIRST TRAINING
#
#################################################################
for param in model_teacher.parameters():
    param.requires_grad = False

for epoch in range(epoch_start, args.epochs):
    
    if patience == max_patience:
        break
    
    train_loss, _, _, train_acc = train_distillation_epoch(model_teacher, model_student, train_loader, cel_criterion, sgd_optimizer, alpha, T, device)
    losses.append(train_loss.item())
    train_accs.append(train_acc)

    if val_loader:

        model_student.train(False)
        val_loss, _, _, val_acc, val_acc_top5, stats = evaluate_distillation(model_teacher, model_student, val_loader, cel_criterion, alpha, T, device)
        model_student.train(True)
        val_accs.append(val_acc)
        val_accs_top5.append(val_acc_top5)
        wandb.log({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_top5_acc': val_acc_top5,
            'val_loss':val_loss,
            'epoch': epoch
        })
    patience = patience + 1
    # Save checkpoints if they are best in the current subset
    if args.save_checkpoints:
        if val_acc > top_val_acc:

            stats = {val_set.inv_dict_labels_dataset[k]:v for k,v in stats.items() if k < args.new_num_classes}
            
            df_stats = pd.DataFrame(stats.items(), columns=['clase', 'Aciertos_Total'])
            df_stats[['Aciertos', 'Total']] = pd.DataFrame(df_stats['Aciertos_Total'].tolist(), index=df_stats.index)
            df_stats.drop(columns=['Aciertos_Total'], inplace=True)
            df_stats['Accuracy'] = df_stats['Aciertos'] / df_stats['Total']
            
            print(df_stats)

            patience = 0

            top_val_acc = val_acc

            model_save_folder_path = "out-checkpoints/" + args.experiment_name

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_student.state_dict(),
                'optimizer_state_dict': sgd_optimizer.state_dict(),
                'loss': train_loss
            }, model_save_folder_path + f'/checkpoint_model.pth')
            
            generate_csv_result(run, model_student, val_loader, model_save_folder_path, val_set.inv_dict_labels_dataset, device)
            generate_csv_accuracy(df_stats, model_save_folder_path)

            artifact = wandb.Artifact(f'best-model_{args.model_type}_{args.prev_num_classes}_{args.new_num_classes}_{run.id}.pth', type='model')
            artifact.add_file(model_save_folder_path + f'/checkpoint_model.pth')
            run.log_artifact(artifact)
            wandb.save(model_save_folder_path + f'/checkpoint_model.pth')

            checkpoint_index += 1


    if epoch % args.log_freq == 0:
        print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item()) + " acc: " + str(train_acc))

        if val_loader:
            print("[" + str(epoch + 1) + "] VALIDATION  loss: " + str(val_loss.item()) + " acc: " + str(val_acc) + " top-5(acc): " + str(val_acc_top5))
        print("Patience:",patience)
        print("")

    # Reset the top accuracies on static subsets
    #if epoch % 10 == 0:
    #    top_train_acc, top_val_acc, val_acc_top5 = 0, 0, 0
    #    checkpoint_index += 1

    lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

#################################################################
#
# SECOND TRAINING
#
#################################################################


'''
for epoch in range(epoch_start, args.epochs):
    
    if patience == max_patience:
        break
    
    train_loss, _, _, train_acc = train_epoch(model, train_loader, cel_criterion, sgd_optimizer, device)
    losses.append(train_loss.item())
    train_accs.append(train_acc)

    if val_loader:
        model.train(False)
        val_loss, _, _, val_acc, val_acc_top5 = evaluate(model, val_loader, cel_criterion, device)
        model.train(True)
        val_accs.append(val_acc)
        val_accs_top5.append(val_acc_top5)
        wandb.log({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_top5_acc': val_acc_top5,
            'val_loss':val_loss,
            'epoch': epoch
        })
    patience = patience + 1
    # Save checkpoints if they are best in the current subset
    if args.save_checkpoints:
        if val_acc > top_val_acc:

            patience = 0

            top_val_acc = val_acc

            model_save_folder_path = "out-checkpoints/" + args.experiment_name

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': sgd_optimizer.state_dict(),
                'loss': train_loss
            }, model_save_folder_path + f'/checkpoint_{args.model_type}_model.pth')
            
            generate_csv_result(run, model, val_loader, model_save_folder_path, val_set.inv_dict_labels_dataset, device)

            artifact = wandb.Artifact(f'best-model_{args.prev_num_classes}_{args.new_num_classes}_{run.id}.pth', type='model')
            artifact.add_file(model_save_folder_path + f'/checkpoint_{args.model_type}_model.pth')
            run.log_artifact(artifact)
            wandb.save(model_save_folder_path + f'/checkpoint_{args.model_type}_model.pth')

            checkpoint_index += 1


    if epoch % args.log_freq == 0:
        print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item()) + " acc: " + str(train_acc))

        if val_loader:
            print("[" + str(epoch + 1) + "] VALIDATION  loss: " + str(val_loss.item()) + " acc: " + str(val_acc) + " top-5(acc): " + str(val_acc_top5))
        print("Patience:",patience)
        print("")

    # Reset the top accuracies on static subsets
    #if epoch % 10 == 0:
    #    top_train_acc, top_val_acc, val_acc_top5 = 0, 0, 0
    #    checkpoint_index += 1

    lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

'''