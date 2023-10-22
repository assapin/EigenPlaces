
import argparse
import os


def append_sm_args(args):
    args.train_dataset_folder = os.environ.get("SM_CHANNEL_TRAIN", args.train_dataset_folder)
    args.val_dataset_folder = os.environ.get("SM_CHANNEL_VALIDATION", args.val_dataset_folder)
    args.test_dataset_folder = os.environ.get("SM_CHANNEL_TEST", args.test_dataset_folder)
    args.save_dir = os.environ.get("SM_MODEL_DIR", args.save_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_on_sm", type=bool, default=False,
                        help="Is this job running on sagemaker")
    parser.add_argument("--wandb", type=str, default=None,
                        help="name of wandb project to use")
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=15, help="_")
    parser.add_argument("--N", type=int, default=3, help="_")
    parser.add_argument("--focal_dist", type=int, default=10, help="_")  # done GS
    parser.add_argument("--s", type=float, default=100, help="_")
    parser.add_argument("--m", type=float, default=0.4, help="_")
    parser.add_argument("--lambda_lat", type=float, default=1., help="_")
    parser.add_argument("--lambda_front", type=float, default=1., help="_")
    parser.add_argument("--groups_num", type=int, default=0,
                        help="If set to 0 use N*N groups")

    parser.add_argument("--min_images_per_class", type=int, default=5, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152", "dinov2_vitb14"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=40, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=5000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    parser.add_argument("--image_size_dimension", type=int, default=512)
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model_ to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    parser.add_argument("--visualize_classes", type=int, default=0,
                        help="Save map visualizations for X classes in the save_dir")
    
    # Paths parameters
    parser.add_argument("--train_dataset_folder", type=str, default=None,
                        help="path of the folder with training images")
    parser.add_argument("--val_dataset_folder", type=str, default=None,
                        help="path of the folder with val images (split in database/queries)")
    parser.add_argument("--test_dataset_folder", type=str, default=None,
                        help="path of the folder with test images (split in database/queries)")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
    
    args  = parser.parse_args()
    if args.groups_num == 0:
        args.groups_num = args.N * args.N
    if args.run_on_sm:
        args = append_sm_args(args)
    return args
