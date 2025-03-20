import argparse
import logging
import os
import torch
from torch import distributed
import sys
sys.path.append('..')
from utils.utils_logging import init_testing
from utils.utils_callbacks import CallBackVerification
from torch.utils.tensorboard import SummaryWriter
from src import get_model
import os
from utils.utils_config import get_config

def main(args):
    # Initialize distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        distributed.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12585",
            rank=rank,
            world_size=world_size,
        )

    torch.cuda.set_device(local_rank)
    
    # Initialize logging
    path = args.model_path
    parent_dir = os.path.dirname(path)

    os.makedirs(f"{parent_dir}/{args.output}", exist_ok=True)
    init_testing(rank, f"{parent_dir}/{args.output}")
    
    # Initialize writer for logging (only for rank 0)
    summary_writer = SummaryWriter(log_dir=os.path.join(f"{parent_dir}/{args.output}", "tensorboard")) if rank == 0 else None

    backbone = get_model(

        args.backbone,

        num_features=args.num_embeddings,

    ).cuda()
    
    # Load pre-trained weights with weights_only=True
    state_dict = torch.load(args.model_path, map_location='cuda', weights_only=True)
    backbone.load_state_dict(state_dict)
    
    # Wrap model in DDP
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True
    )
    
    backbone.eval()

    # Initialize verification callback
    callback_verification = CallBackVerification(
        val_targets=['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'cfp_ff'],
        # val_targets=['lfw'],
        rec_prefix=args.rec_prefix,
        summary_writer=summary_writer,
        image_size=(112, 112)
    )
    
    # Run verification
    logging.info(f"Starting verification with {args.backbone} ...")
    with torch.no_grad():
        callback_verification(0, backbone)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EdgeFace LFW Verification')
    parser.add_argument('--backbone', type=str, required=True, help='Backbone architecture')
    parser.add_argument('--num_embeddings', type=int, required=True, help='Number of embeddings')
    parser.add_argument('--model_path', type=str, required=True, help='Path to your trained model.pt')
    parser.add_argument('--rec-prefix', type=str, required=True, help='Directory containing lfw.bin')
    parser.add_argument('--output', type=str, default='verification_output', help='Output directory')
    main(parser.parse_args())