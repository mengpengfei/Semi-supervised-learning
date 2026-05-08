from torchvision import transforms

import semilearn
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, \
    BasicDataset
import numpy as np

from semilearn.datasets.augmentation import RandAugment

if __name__ == '__main__':
    # define configs and create config
    config = {
        'algorithm': 'freematch',
        'save_name': 'freematch_cifar10_200_0',
        'net': 'vit_tiny_patch2_32',
        'use_pretrain': True,
        'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',

        # optimization configs
        'epoch': 200,
        'num_train_iter': 204800,
        'num_eval_iter': 256,
        'num_log_iter': 2048,
        'optim': 'AdamW',
        'lr': 5e-4,
        'layer_decay': 0.5,
        'batch_size': 4,
        'eval_batch_size': 4,
        # 'num_warmup_iter': 5120,

        # dataset configs
        'dataset': 'cifar10',
        'num_labels': 200,
        'num_classes': 10,
        'img_size': 32,
        'crop_ratio': 0.875,
        'data_dir': './data',
        'ulb_samples_per_class': None,
        'T': 0.5,
        'ema_p': 0.999,
        'ent_loss_ratio': 0.001,
        'use_cat': True,

        # algorithm specific configs
        'hard_label': True,
        'uratio': 1,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 0,
        'world_size': 1,
        'distributed': False,
        "num_workers": 2,
        "amp": False
    }
    config = get_config(config)

    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
    dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir, include_lb_to_ulb=config.include_lb_to_ulb)
    train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
    train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
    eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size,data_sampler=None)
    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)

    trainer.evaluate(eval_loader)
    y_pred, y_logits = trainer.predict(eval_loader)

    # # create model and specify algorithm
    # algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
    # # replace with your own code
    # data = np.random.randint(0, 255, size=3072 * 1000).reshape((-1, 32, 32, 3))
    # data = np.uint8(data)
    # target = np.random.randint(0, 10, size=1000)
    # lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(config, data, target,
    #                                                           lb_num_labels=config.num_labels,
    #                                                           num_classes=config.num_classes)
    #
    # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomCrop(32, padding=int(32 * 0.125), padding_mode='reflect'),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #
    # strong_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                        transforms.RandomCrop(32, padding=int(32 * 0.125), padding_mode='reflect'),
    #                                        RandAugment(3, 5),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #
    #
    # lb_dataset = BasicDataset(config.algorithm, lb_data, lb_target, config.num_classes, train_transform, is_ulb=False)
    # ulb_dataset = BasicDataset(config.algorithm, ulb_data, ulb_target, config.num_classes, train_transform, is_ulb=True, strong_transform=strong_transform)
    #
    #
    # # replace with your own code
    # eval_data = np.random.randint(0, 255, size=3072 * 100).reshape((-1, 32, 32, 3))
    # eval_data = np.uint8(eval_data)
    # eval_target = np.random.randint(0, 10, size=100)
    #
    # eval_transform = transforms.Compose([transforms.Resize(32),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #
    # eval_dataset = BasicDataset(config.algorithm, lb_data, lb_target, config.num_classes, eval_transform, is_ulb=False)
    #
    # # define data loaders
    # train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size)
    # train_ulb_loader = get_data_loader(config, ulb_dataset, int(config.batch_size * config.uratio))
    # eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size)
    #
    # # training and evaluation
    # trainer = Trainer(config, algorithm)
    # trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
    # trainer.evaluate(eval_loader)




