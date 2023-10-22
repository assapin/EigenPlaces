
import sys
import torch
import logging
import torchmetrics
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as tfm

import test
import util
import parser
import commons
import cosface_loss
import augmentations
from eigenplaces_model import eigenplaces_network
from datasets.test_dataset import TestDataset
from datasets.eigenplaces_dataset import EigenPlacesDataset
import wandb

def log_model_to_wandb(local_path, name):
    artifact = wandb.Artifact(name, type='model', description='trained model')
    artifact.add_file(local_path)
    wandb.log_artifact(artifact)

def train():

    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = parser.parse_arguments()
    if args.wandb:
        wandb.init(project=args.wandb, config=args)
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    #### Model
    model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device).train()

    #### Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #### Datasets
    groups = [EigenPlacesDataset(
        args.train_dataset_folder, resize = args.image_size_dimension, M=args.M, N=args.N, focal_dist=args.focal_dist,
        current_group=n // 2, min_images_per_class=args.min_images_per_class,
        angle=[0, 90][n % 2], visualize_classes=args.visualize_classes)
        for n in range(args.groups_num * 2)
    ]
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [cosface_loss.MarginCosineProduct(
        args.fc_output_dim, len(group), s=args.s, m=args.m) for group in groups]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                              classifiers]

    gpu_augmentation = tfm.Compose([
        augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                contrast=args.contrast,
                                                saturation=args.saturation,
                                                hue=args.hue),
        augmentations.DeviceAgnosticRandomResizedCrop([args.image_size_dimension, args.image_size_dimension],
                                                      scale=[1 - args.random_resized_crop, 1]),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following "
                 "number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following "
                 "number of images {[g.get_images_num() for g in groups]}")

    logging.info(f"There are {len(groups[0])} classes for the first group, " +
                 f"each epoch has {args.iterations_per_epoch} iterations " +
                 f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                 f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

    val_ds = TestDataset(f"{args.val_dataset_folder}", queries_folder="queries")
    logging.info(f"Validation set: {val_ds}")

    #### Resume
    if args.resume_train:
        model, model_optimizer, classifiers, classifiers_optimizers, \
            best_val_recall1, start_epoch_num = \
            util.resume_train(args, output_folder, model, model_optimizer,
                              classifiers, classifiers_optimizers)

        model = model.to(args.device)
        epoch_num = start_epoch_num - 1
        logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} " +
                     f"from checkpoint {args.resume_train}")
    else:
        best_val_recall1 = start_epoch_num = 0

    #### Train / evaluation loop
    logging.info("Start training ...")

    scaler = torch.cuda.amp.GradScaler()

    for epoch_num in range(start_epoch_num, args.epochs_num):

        #### Train
        epoch_start_time = datetime.now()

        def get_iterator(groups, classifiers, classifiers_optimizers, batch_size, g_num):
            assert len(groups) == len(classifiers) == len(classifiers_optimizers)
            classifiers[g_num] = classifiers[g_num].to(args.device)
            util.move_to_device(classifiers_optimizers[g_num], args.device)
            return commons.InfiniteDataLoader(groups[g_num], num_workers=args.num_workers,
                                              batch_size=batch_size, shuffle=True,
                                              pin_memory=(args.device == "cuda"), drop_last=True)

        # Select classifier and dataloader according to epoch
        current_dataset_num = (epoch_num % args.groups_num) * 2

        iterators = []
        for i in range(2):
            iterators.append(get_iterator(groups, classifiers, classifiers_optimizers,
                                          args.batch_size, current_dataset_num + i))
        lateral_loss = torchmetrics.MeanMetric()
        frontal_loss = torchmetrics.MeanMetric()

        model = model.train()
        if args.wandb:
            wandb.watch(model)

        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
            model_optimizer.zero_grad()

            #### EigenPlace ITERATION ####
            for i in range(2):
                classifiers_optimizers[current_dataset_num + i].zero_grad()

                images, targets, _ = next(iterators[i])
                images, targets = images.to(args.device), targets.to(args.device)
                with torch.cuda.amp.autocast():
                    images = gpu_augmentation(images)
                    descriptors = model(images)
                    output = classifiers[current_dataset_num + i](descriptors, targets)
                    loss = criterion(output, targets)
                    if i == 0:
                        loss *= args.lambda_lat
                    else:
                        loss *= args.lambda_front
                del images, output
                scaler.scale(loss).backward()
                scaler.step(classifiers_optimizers[current_dataset_num + i])
                if i == 0:
                    lateral_loss.update(loss.detach().cpu())
                else:
                    frontal_loss.update(loss.detach().cpu())
                del loss
                #######################

            scaler.step(model_optimizer)
            scaler.update()

        for i in range(2):
            classifiers[current_dataset_num + i] = classifiers[current_dataset_num + i].cpu()
            util.move_to_device(classifiers_optimizers[current_dataset_num + i], "cpu")

        logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]} - "
                      f"group {current_dataset_num} lateral_loss = {lateral_loss.compute():.4f} - "
                      f"group {current_dataset_num + 1} frontal_loss = {frontal_loss.compute():.4f}")

        #### Evaluation
        recalls, recalls_str = test.test(args, val_ds, model, batchify=True, resize=args.image_size_dimension or None)
        logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str}")

        if args.wandb:
            wandb.log({
                'Validation_Lateral_Loss': lateral_loss.compute(),
                'Validation_Frontal_Loss': frontal_loss.compute(),
                'Validation_Recall@1': recalls[0]
            })
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        # Save checkpoint, which contains all training parameters
        checkpoint, best_model_path = util.save_checkpoint({
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "classifiers_state_dict": [c.state_dict() for c in classifiers],
            "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
            "best_val_recall1": best_val_recall1
        }, is_best, output_folder)
        if args.wandb:
            log_model_to_wandb(checkpoint, name=f"model_epoch_{epoch_num + 1}")
            if best_model_path:
                log_model_to_wandb(best_model_path, name=f"best_model_epoch_{epoch_num + 1}")


    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    #### Test best model_ on test set v1
    best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
    model.load_state_dict(best_model_state_dict)

    test_ds = TestDataset(f"{args.test_dataset_folder}", queries_folder="queries_v1")
    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")
    if args.wandb:
        wandb.log({
            'Test_Recall@1': recalls[0]
        })

        logging.info("Experiment finished (without any errors)")

        # 3. Close wandb
        wandb.finish()

    logging.info("Experiment finished (without any errors)")


if __name__ == '__main__':
    train()

