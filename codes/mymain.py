import os
import os.path as osp
import math
import argparse
import yaml
import time

import torch   ###########################SERVE ANCHE IL TRAIN? COME SI FA?


from data import mycreate_dataloader, prepare_data
from models import define_model
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import base_utils, data_utils


def test(opt):
    # logging
    logger = base_utils.get_logger('base')
    if opt['verbose']:
        logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
        base_utils.print_options(opt, logger)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # setup model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]

        # log
        logger.info('=' * 40)
        logger.info('Testing model: {}'.format(model_idx))
        logger.info('=' * 40)

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)

        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # use dataset with prefix `test`
            if not dataset_idx.startswith('test'):
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            logger.info('Testing on {}: {}'.format(dataset_idx, ds_name))

            # define metric calculator
            try:
                metric_calculator = MetricCalculator(opt)
            except:
                print('No metirc need to compute!')

            # create data loader
            test_loader = create_dataloader(opt, dataset_idx=dataset_idx)

            # infer and store results for each sequence
            for i, data in enumerate(test_loader):

                # fetch data
                lr_data = data['lr'][0]
                lr_data1=data['lr1'][0]
                seq_idx = data['seq_idx'][0]
                frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

                # infer
                hr_seq = model.infer(lr_data,lr_data1)  # thwc|rgb|uint8

                # save results (optional)
                if opt['test']['save_res']:
                    res_dir = osp.join(opt['test']['res_dir'], ds_name, model_idx)
                    res_seq_dir = osp.join(res_dir, seq_idx)
                    data_utils.save_sequence(res_seq_dir, hr_seq, frm_idx, to_bgr=True)

                # compute metrics for the current sequence ####GT_SEQ DOVREBBE FORSE ESSERE UNA SOLA, QUELLA VERA NON QUELLA SEPARATA
                true_seq_dir = osp.join(opt['dataset'][dataset_idx]['gt_seq_dir'], seq_idx)
                try:
                    metric_calculator.compute_sequence_metrics(seq_idx, true_seq_dir, '', pred_seq=hr_seq)
                except:
                    print('No metirc need to compute!')

            # save/print metrics
            try:
                if opt['test'].get('save_json'):
                    # save results to json file
                    json_path = osp.join(
                        opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                    metric_calculator.save_results(model_idx, json_path, override=True)
                else:
                    # print directly
                    metric_calculator.display_results()

            except:
                print('No metirc need to save!')

            logger.info('-' * 40)

    # logging
    logger.info('Finish testing')
    logger.info('=' * 40)