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
from spoter.utils import train_epoch, evaluate
from spoter.utils import train_distillation_epoch, evaluate_distillation
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

dict_input = args.word_list_path
version = int(os.path.splitext(os.path.basename(dict_input))[0].split('_')[-1][1:])
print(version) 

#args.model_type = "simple"#"simple"#"fixed_linear"
limit_type = args.limit_type #"fixed_with_old" #"fixed"
#previous_model_type = "simple"#"simple"#"fixed_linear"#"base"
maximun_train = args.maximun_train
maximun_val = args.maximun_val
instance_inc = args.instance_inc
increment_count = args.increment_count

df_words = pd.read_csv(args.word_list_path, encoding='utf-8', header=None)
words = list(df_words[0])

inc_range = args.increment_count.split('-')
inc_range = [int(_i) for _i in inc_range]

if len(inc_range) > 1:
    args.prev_num_classes  = inc_range[-2]
    args.new_num_classes = inc_range[-1]
else:
    args.prev_num_classes, args.new_num_classes = inc_range[0], inc_range[0]

all_words = words[:args.new_num_classes]
new_words = words[args.prev_num_classes:args.new_num_classes]
old_words = words[:args.prev_num_classes]

max_patience = 200
alpha = args.prev_num_classes / args.new_num_classes # old / total
T = 2

dataset_name = args.training_set_path.split("--")[0].split('/')[-1]

if args.previous_model_type == "Base" or args.previous_model_type == 'Fixed_Base':
    args.load_model_from = f"{args.previous_model_type}_{dataset_name}_spoter2_{args.prev_num_classes}_{args.prev_num_classes}_V{version}/"
else:
    args.load_model_from = f"{args.previous_model_type}_{dataset_name}_{limit_type}_spoter2_{inc_range[-3]}_{args.prev_num_classes}_V{version}/"
    #if limit_type == 'NIC':
    #    args.load_model_from = f"{args.previous_model_type}_{dataset_name}_old_spoter_{inc_range[-3]}_{args.prev_num_classes}_V{version}/"
#args.epochs = 1000
#args.lr = 0.00005

PROJECT_WANDB = "ISAAC_incremental_learning" #ISAAC_incremental_learning #SIMBig_incremental_learning
ENTITY = "joenatan30" 
TAG = [f'prev_{args.prev_num_classes}',f'new_{args.new_num_classes}', args.model_type, f'V{version}', dataset_name]

if args.model_type == "Base" or args.model_type == 'Fixed_Base':
    args.experiment_name = f'{args.model_type}_{dataset_name}_spoter2_{args.prev_num_classes}_{args.prev_num_classes}_V{version}'
else:
    args.experiment_name = f'{args.model_type}_{dataset_name}_{limit_type}_spoter2_{args.prev_num_classes}_{args.new_num_classes}_V{version}'

run = wandb.init(project=PROJECT_WANDB,
                 entity=ENTITY,
                 config=args,
                 name=args.experiment_name,
                 job_type="model-training",
                 tags=TAG)

config = wandb.config
wandb.watch_called = False

if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
    model_teacher, model_student = incremental_model_type(args)
else:
    model = incremental_model_type(args)

# DATA LOADER
# Training set
transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
if args.prev_num_classes == args.new_num_classes:
    train_set = LSP_Dataset(args.training_set_path, words=old_words, transform=transform, have_aumentation=True, keypoints_model='mediapipe',
                        limit_type=limit_type, instance_inc=instance_inc, increment_count=increment_count)
elif limit_type == "NIC" or limit_type=='exemplar':
    train_set = LSP_Dataset(args.training_set_path, words=all_words, transform=transform, have_aumentation=True, keypoints_model='mediapipe',
                        limit_type=limit_type, instance_inc=instance_inc, increment_count=increment_count)
else:
    train_set = LSP_Dataset(args.training_set_path, words=new_words, transform=transform, have_aumentation=True, keypoints_model='mediapipe',
                        limit_type=limit_type, instance_inc=instance_inc, increment_count=increment_count)

# Validation set
val_set = LSP_Dataset(args.validation_set_path, words=all_words, keypoints_model='mediapipe', have_aumentation=False,
                      limit_type="NC", instance_inc=maximun_val, increment_count=increment_count)
val_loader = DataLoader(val_set, shuffle=False, generator=g)

# Testing set
if args.testing_set_path:
    eval_set = LSP_Dataset(args.testing_set_path, words=words, keypoints_model='mediapipe')
    eval_loader = DataLoader(eval_set, shuffle=False, generator=g)

else:
    eval_loader = None

train_loader = DataLoader(train_set, shuffle=True, generator=g)

#class_weight = torch.FloatTensor([1/train_set.label_freq[i] for i in range(args.new_num_classes)]).to(device)
cel_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)#, weight=class_weight)

if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
    sgd_optimizer = optim.SGD(model_student.parameters(), lr=args.lr)
else:
    sgd_optimizer = optim.SGD(model.parameters(), lr=args.lr)

epoch_start = 0

# Ensure that the path for checkpointing and for images both exist
Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
Path("out-img/").mkdir(parents=True, exist_ok=True)


print("#"*50)
print("#"*30)
print("#"*10)
if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
    print("Num Trainable Params: ", sum(p.numel() for p in model_student.parameters() if p.requires_grad))
else:
    print("Num Trainable Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("#"*10)
print("#"*30)
print("#"*50)


# MARK: TRAINING
train_acc, val_acc = 0, 0
losses, train_accs, val_accs, val_accs_top5 = [], [], [], []
lr_progress = []
top_train_acc, top_val_acc = 0, 0
checkpoint_index = 0

if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
    model_student.train(True)
    model_student.to(device)
    model_teacher.to(device)

else:
    model.train(True)
    model.to(device)

patience = 0

for epoch in range(epoch_start, args.epochs):

    if patience == max_patience:
        break
    
    if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
        train_loss, _, _, train_acc = train_distillation_epoch(model_teacher, model_student, train_loader, cel_criterion, sgd_optimizer, alpha, T, device) 
    else:
        train_loss, _, _, train_acc = train_epoch(model, train_loader, cel_criterion, sgd_optimizer, device)
        
    losses.append(train_loss.item())
    train_accs.append(train_acc)

    if val_loader:

        if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
            model_student.train(False)
            val_loss, _, _, val_acc, val_acc_top5, stats = evaluate_distillation(model_teacher, model_student, val_loader, cel_criterion, alpha, T, device)
            model_student.train(True)
        else:
            model.train(False)
            val_loss, _, _, val_acc, val_acc_top5, stats = evaluate(model, val_loader, cel_criterion, device)
            model.train(True)

        val_accs.append(val_acc)
        val_accs_top5.append(val_acc_top5)
        wandb.log({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_top5_acc': val_acc_top5,
            'val_loss':val_loss,
            'epoch': epoch,
            "top_val_acc": top_val_acc,
        })
    patience = patience + 1
    # Save checkpoints if they are best in the current subset
    if args.save_checkpoints:
        if val_acc > top_val_acc:
            #print(stats)
            print(val_set.inv_dict_labels_dataset)
            stats = {val_set.inv_dict_labels_dataset[k]:v for k,v in stats.items() if k < args.new_num_classes}
            
            df_stats = pd.DataFrame(stats.items(), columns=['clase', 'Aciertos_Total'])
            df_stats[['Aciertos', 'Total']] = pd.DataFrame(df_stats['Aciertos_Total'].tolist(), index=df_stats.index)
            df_stats.drop(columns=['Aciertos_Total'], inplace=True)
            df_stats['Accuracy'] = df_stats['Aciertos'] / df_stats['Total']
            
            print(df_stats)

            patience = 0

            top_val_acc = val_acc

            model_save_folder_path = "out-checkpoints/" + args.experiment_name

            if args.model_type == "Weighted" or args.model_type == 'Weighted_frozen':
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_student.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss
                }, model_save_folder_path + f'/checkpoint_model.pth')
                generate_csv_result(run, model_student, val_loader, model_save_folder_path, val_set.inv_dict_labels_dataset, device)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss
                }, model_save_folder_path + f'/checkpoint_model.pth')
                generate_csv_result(run, model, val_loader, model_save_folder_path, val_set.inv_dict_labels_dataset, device)
            
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

# MARK: TESTING
'''
print("\nTesting checkpointed models starting...\n")

top_result, top_result_name = 0, ""

if eval_loader:
    for i in range(checkpoint_index):
        for checkpoint_id in ["v"]: #["t", "v"]:
            # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
            tested_model = torch.load("out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
            tested_model.train(False)
            _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

            if eval_acc > top_result:
                top_result = eval_acc
                top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

            print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

    print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
  
  '''