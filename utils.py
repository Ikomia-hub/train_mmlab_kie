import os.path
import random
import numpy as np
import json
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import LoggerHook
from mmcv.runner.dist_utils import master_only
from mmocr.datasets.openset_kie_dataset import OpensetKIEDataset
import torch
from mmdet.datasets.builder import DATASETS
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets.kie_dataset import KIEDataset
import os.path as osp
import warnings


class UserStop(Exception):
    pass


def register_mmlab_modules():
    # Define custom hook to stop process when user uses stop button and to save last checkpoint

    @HOOKS.register_module(force=True)
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emitStepProgress):
            self.stop = stop
            self.output_folder = output_folder
            self.emitStepProgress = emitStepProgress

        def after_epoch(self, runner):
            self.emitStepProgress()

        def after_train_iter(self, runner):
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomMlflowLoggerHook(LoggerHook):
        """Class to log metrics and (optionally) a trained model to MLflow.
        It requires `MLflow`_ to be installed.
        Args:
            interval (int): Logging interval (every k iterations). Default: 10.
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`. Default: True.
            reset_flag (bool): Whether to clear the output buffer after logging.
                Default: False.
            by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """

        def __init__(self,
                     log_metrics,
                     interval=10,
                     ignore_last=True,
                     reset_flag=False,
                     by_epoch=False):
            super(CustomMlflowLoggerHook, self).__init__(interval, ignore_last,
                                                         reset_flag, by_epoch)
            self.log_metrics = log_metrics

        @master_only
        def log(self, runner):
            tags = self.get_loggable_tags(runner)
            if tags:
                self.log_metrics(tags, step=self.get_iter(runner))

    @DATASETS.register_module(force=True)
    class MyOpensetKIEDataset(OpensetKIEDataset):
        key_node_idx = 1

        def __init__(self,
                     ann_file,
                     loader,
                     dict_file,
                     img_prefix='',
                     pipeline=None,
                     norm=10.,
                     link_type='one-to-one',
                     edge_thr=0.5,
                     test_mode=True,
                     key_node_idx=0,
                     value_node_idx=1,
                     node_classes=4):
            super().__init__(ann_file, loader, dict_file, img_prefix, pipeline,
                             norm, link_type, edge_thr, test_mode, key_node_idx, value_node_idx, node_classes)
            with open(dict_file, 'r') as f:
                lines = f.readlines()

            dict_list = ""
            for line in lines:
                char = line.rstrip("\n")
                if char == "":
                    char = " "
                dict_list += char
            self.dict = {
                '': 0,
                **{
                    line.rstrip('\r\n'): ind
                    for ind, line in enumerate(dict_list, 1)
                }
            }

    @DATASETS.register_module(force=True)
    class MyKIEDataset(KIEDataset):
        """
        Args:
            ann_file (str): Annotation file path.
            pipeline (list[dict]): Processing pipeline.
            loader (dict): Dictionary to construct loader
                to load annotation infos.
            img_prefix (str, optional): Image prefix to generate full
                image path.
            test_mode (bool, optional): If True, try...except will
                be turned off in __getitem__.
            dict_file (str): Character dict file path.
            norm (float): Norm to map value from one range to another.
        """

        def __init__(self,
                     ann_file=None,
                     loader=None,
                     dict_file=None,
                     img_prefix='',
                     pipeline=None,
                     norm=10.,
                     directed=False,
                     test_mode=True,
                     **kwargs):
            if ann_file is None and loader is None:
                warnings.warn(
                    'KIEDataset is only initialized as a downstream demo task '
                    'of text detection and recognition '
                    'without an annotation file.', UserWarning)
            else:
                super().__init__(
                    ann_file=ann_file,
                    loader=loader,
                    pipeline=pipeline,
                    dict_file=dict_file,
                    img_prefix=img_prefix,
                    test_mode=test_mode)
                assert osp.exists(dict_file)

            with open(dict_file, 'r') as f:
                lines = f.readlines()

            dict_list = ""
            for line in lines:
                char = line.rstrip("\n")
                if char == "":
                    char = " "
                dict_list += char
            self.dict = {
                '': 0,
                **{
                    line.rstrip('\r\n'): ind
                    for ind, line in enumerate(dict_list, 1)
                }
            }


def prepare_dataset(ikdataset, split, output_dataset):
    n_img = len(ikdataset['images'])
    n_train = int(split * n_img)
    # Indices of train images
    idx_train = random.sample(list(np.arange(n_img)), n_train)
    train_text_file = os.path.join(output_dataset, "train.txt")
    test_text_file = os.path.join(output_dataset, "test.txt")
    rewrite = True
    if not (os.path.isdir(output_dataset)):
        os.mkdir(output_dataset)
    if os.path.isfile(train_text_file):
        # check the number of lines in the train.txt to know if dataset must be rewritten
        with open(train_text_file, 'r') as f:
            data = f.readlines()
            rewrite = n_train != len(data)
    # KIE dataset can be on closeset format or openset format
    # https://mmocr.readthedocs.io/en/latest/tutorials/kie_closeset_openset.html
    openset = False
    # If train.txt has not as many lines as intended, the function rewrites train.txt and test.txt to fit the split
    # ratio
    if rewrite:
        print("Preparing dataset...")
        with open(train_text_file, 'w') as f:
            f.write('')
        with open(test_text_file, 'w') as f:
            f.write('')

        for i, img in enumerate(ikdataset['images']):
            if i in idx_train:
                file_to_write = train_text_file
            else:
                file_to_write = test_text_file
            record = {}
            record["file_name"] = img["filename"]
            record["height"] = img["height"]
            record["width"] = img["width"]
            record["annotations"] = []
            for annot in img["annotations"]:
                sample = {}
                sample['label'] = annot["category_id"]
                sample['text'] = annot['text']
                if 'bbox' in annot:
                    x, y, w, h = annot['bbox']
                    sample['box'] = [x, y, x + w, y, x + w, y + h, x, y + h]
                else:
                    sample['box'] = annot['segmentation_poly'][0]
                if 'edge' in annot:
                    sample['edge'] = annot["edge"]
                    openset = True
                record["annotations"].append(sample)
            dict_to_write = json.dumps(record)
            with open(file_to_write, 'a') as f:
                f.write(dict_to_write + '\n')
        print("Dataset prepared!")
        return openset
    else:
        for line in data:
            sample = json.loads(line)
            for annot in sample["annotations"]:
                if 'edge' in annot.keys():
                    return True
                else:
                    return False
        return False
    raise Exception
