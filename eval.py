
import sys
import wandb
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from datasets.test_dataset import TestDataset
from eigenplaces_model import eigenplaces_network

def eval():
    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = parser.parse_arguments()
    if args.wandb:
        wandb.init(project=args.wandb, config=args)

    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    #### Model
    model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.info(f"Loading model_ from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)
    else:
        logging.info("WARNING: You didn't provide a path to resume the model_ (--resume_model parameter). " +
                     "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)
    if args.wandb:
        wandb.watch(model)

    test_ds = TestDataset(f"{args.test_dataset_folder}", queries_folder="queries_v1")
    recalls, recalls_str = test.test(args=args, eval_ds=test_ds, model=model, resize=args.image_size_dimension)
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
    eval()



