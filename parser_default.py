import argparse

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter", help="Name of the experiment after which the logs and plots will be named")
    #parser.add_argument("--prev_num_classes", type=int, default=5, help="Number of classes to be recognized by the model")
    #parser.add_argument("--new_num_classes", type=int, default=10, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108, help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=1, help="Seed with which to initialize all the random components of the training")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1, help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True, help="Determines whether to save weights checkpoints")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001, help="Standard deviation parameter for Gaussian noise layer")


    ####

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")

    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")
    parser.add_argument("--validation_set_size", type=float, help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                                                                  " to 'split-from-train'")

    parser.add_argument("--device", type=int, default=0, help="Determines which Nvidia device will use (just one number)")
    
    parser.add_argument('--word_list_path', type=str, required=True, help='relative path of scv of word list.')
    parser.add_argument('--model_type', required=True, choices=['Fixed', 'Expanded', 'Weighted', 'Base','Fixed_Base','Expanded_frozen', 'Weighted_frozen', 'Fixed_frozen'], help='Elija una opción entre Fixed, Expanded y Weighted')
    parser.add_argument('--previous_model_type', required=True, choices=['Fixed', 'Expanded', 'Weighted', 'Base','Fixed_Base', 'Expanded_frozen', 'Weighted_frozen', 'Fixed_frozen'], help='Elija una opción entre Fixed, Expanded y Weighted')
    parser.add_argument('--limit_type', required=True, choices=['NC','NIC','exemplar'])
    
    parser.add_argument('--instance_inc', type=int, required=True)
    parser.add_argument('--increment_count', type=str, required=True)

    parser.add_argument('--dataset_reference', required=True, type=int)
    #parser.add_argument('--prev_num_classes', required=True, type=int)
    #parser.add_argument('--new_num_classes', required=True, type=int)

    parser.add_argument('--maximun_train', type=int)
    parser.add_argument('--maximun_val', type=int)


    return argparse.ArgumentParser("", parents=[parser], add_help=False)