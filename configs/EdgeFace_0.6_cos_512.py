from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "EdgeFace"
config.resume = False
config.seed = 42
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.05
config.batch_size = 1700
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 3000
config.dali = True
config.save_all_states = True
config.rank_ratio = 0.6
# config.dali_aug = True

config.num_workers = 64

config.rec = "/home/bac.dx/datasets/shuffled_ms1m-retinaface-t1"
loss = "cos"
config.output = f'results/{config.network}_{config.rank_ratio}_{loss}_{config.embedding_size}'

config.num_classes = 93431
config.num_images = 6179700

config.num_epoch = 50
config.warmup_epoch = 5
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", "calfw", "cplfw", "cfp_ff"]
# config.val_targets = []