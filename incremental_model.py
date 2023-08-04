import torch
import torch.nn as nn

from Incremental_model.simple import simpleSpoter
from Incremental_model.linear import LinearSpoter
from spoter import spoter_model

def incremental_model_type(args):

    if args.prev_num_classes == args.new_num_classes:

        #model = simpleSpoter(args.prev_num_classes)
        model = spoter_model.SPOTER(args.prev_num_classes, hidden_dim=54*2)
        return model

    if args.model_type == "simple":

        #model = simpleSpoter(args.prev_num_classes)
        model = spoter_model.SPOTER(args.prev_num_classes, hidden_dim=54*2)
        # Load pretrained model
        checkpoint = torch.load(f'checkpoint_{args.model_type}_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        pretrained_weights = model.linear_class.weight.data
        pretrained_biases = model.linear_class.bias.data
        model.linear_class = nn.Linear(model.linear_class.in_features, args.new_num_classes)
        model.linear_class.weight.data[: args.prev_num_classes] = pretrained_weights
        model.linear_class.bias.data[: args.prev_num_classes] = pretrained_biases
        '''
        pretrained_weights = model.pretrained_model.linear_class.weight.data
        pretrained_biases = model.pretrained_model.linear_class.bias.data
        model.pretrained_model.linear_class = nn.Linear(model.pretrained_model.linear_class.in_features, args.new_num_classes)
        model.pretrained_model.linear_class.weight.data[: args.prev_num_classes] = pretrained_weights
        model.pretrained_model.linear_class.bias.data[: args.prev_num_classes] = pretrained_biases
        '''
        return model
    elif args.model_type == "distillation":

        #model = simpleSpoter(args.prev_num_classes)
        model_1 = spoter_model.SPOTER(args.prev_num_classes, hidden_dim=54*2)
        model_2 = spoter_model.SPOTER(args.prev_num_classes, hidden_dim=54*2)
        
        checkpoint = torch.load(f'checkpoint_{args.model_type}_model.pth')
        model_1.load_state_dict(checkpoint['model_state_dict'])
        model_2.load_state_dict(checkpoint['model_state_dict'])

        pretrained_weights = model_1.linear_class.weight.data
        pretrained_biases = model_1.linear_class.bias.data
        model_2.linear_class = nn.Linear(model.linear_class.in_features, args.new_num_classes)
        model_2.linear_class.weight.data[: args.prev_num_classes] = pretrained_weights
        model_2.linear_class.bias.data[: args.prev_num_classes] = pretrained_biases

        return model_1, model_2

    
