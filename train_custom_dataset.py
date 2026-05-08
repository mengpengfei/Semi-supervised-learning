import argparse

from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer


from semilearn.datasets.cv_datasets.cbcnet import get_cbcnet

if __name__ == '__main__':
    # define configs and create config
    config_ = {
        'algorithm': 'fullysupervised',
        'save_name': 'fullysupervised_cbcnet_1',
        'net': 'vit_tiny_patch2_32',
        'use_pretrain': False,
        'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',

        # optimization configs
        'epoch': 200,
        'num_train_iter': 4480300,
        'num_eval_iter': 22401,
        'num_log_iter': 200,
        'optim': 'AdamW',
        'lr': 0.0005,
        'layer_decay': 0.5,
        'batch_size': 8,
        'eval_batch_size': 8,


        # dataset configs
        'dataset': 'none',
        # 'num_labels': 179212,
        'num_labels': 179212,
        'num_classes': 4364,
        'img_size': 32,
        'crop_ratio': 0.92,
        'data_dir': '/data2/fssd2/damagou_error',
        # 'data_dir': '/data2/code/Semi-supervised-learning/data/cbcnet',
        'ulb_samples_per_class': None,
        'use_cat': True,

        # algorithm specific configs
        'hard_label': True,
        'uratio': 1,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 0,
        'world_size': 1,
        'distributed': False,
        "num_workers": 4,
        "amp": False
    }

    config = get_config(config_)

    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
    # args.img_size = config.img_size
    # args.crop_ratio=config.crop_ratio
    lb_dset, ulb_dset, eval_dset = get_cbcnet(config, algorithm, name=None, num_labels=config.num_labels, num_classes=config.num_classes, data_dir=config.data_dir)

    # Debug: Print actual dataset sizes
    print(f"Labeled dataset size: {len(lb_dset)}")
    print(f"Unlabeled dataset size: {len(ulb_dset)}")


    # define data loaders
    train_lb_loader = get_data_loader(config, lb_dset, config.batch_size)
    train_ulb_loader = get_data_loader(config, ulb_dset, int(config.batch_size * config.uratio))
    # eval_loader = get_data_loader(config, eval_dset, config.eval_batch_size,data_sampler=None)
    eval_loader = get_data_loader(config, eval_dset, config.eval_batch_size)

    # Debug: Print loader lengths after creation
    print(f"Train LB loader iterations: {len(train_lb_loader)}")
    print(f"Train ULB loader iterations: {len(train_ulb_loader)}")

    # training and evaluation
    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
    trainer.evaluate(eval_loader)




