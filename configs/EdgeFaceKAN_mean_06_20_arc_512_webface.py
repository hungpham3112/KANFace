from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "EdgeFaceKAN_mean"
config.resume = False
config.seed = 42
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 5000
config.dali = True
config.save_all_states = True
config.grid_size = 20
config.rank_ratio = 0.6
# config.dali_aug = True

config.num_workers = 64

config.rec = "/home/bac.dx/datasets/WebFace_260M/webface12m"

loss = "arc"
config.output = f'results/{config.network}_0{int(config.rank_ratio * 10)}_{config.grid_size}_{loss}_{config.embedding_size}'

config.num_classes = 617970
config.num_images = 12720066

config.num_epoch = 50
config.warmup_epoch = 5
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", "calfw", "cplfw", "cfp_ff"]
# config.val_targets = []

