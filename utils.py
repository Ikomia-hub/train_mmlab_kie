import os.path
import random
import numpy as np
import json
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.hooks import LoggerHook
from typing import Optional, Sequence, Union, Dict

DATA_BATCH = Optional[Union[dict, tuple, list]]


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

        def _after_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Union[Sequence, dict]] = None,
                        mode: str = 'train') -> None:
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomMlflowLoggerHook(LoggerHook):
        """Class to log metrics and (optionally) a trained model to MLflow.
        It requires `MLflow`_ to be installed.
        Args:
            log_metrics (function): Logging function.
            interval (int): Logging interval (every k iterations). Default: 10.
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`. Default: True.
            by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """

        def __init__(self,
                     log_metrics,
                     interval=10,
                     ignore_last=True,
                     by_epoch=False):
            super(CustomMlflowLoggerHook, self).__init__(interval=interval, ignore_last=ignore_last,
                                                         log_metric_by_epoch=by_epoch)
            self.log_metrics = log_metrics

        def after_val_epoch(self,
                            runner,
                            metrics: Optional[Dict[str, float]] = None) -> None:
            """All subclasses should override this method, if they need any
            operations after each validation epoch.

            Args:
                runner (Runner): The runner of the validation process.
                metrics (Dict[str, float], optional): Evaluation results of all
                    metrics on validation dataset. The keys are the names of the
                    metrics, and the values are corresponding results.
            """
            tag, log_str = runner.log_processor.get_log_after_epoch(
                runner, len(runner.val_dataloader), 'val')
            runner.logger.info(log_str)
            self.log_metrics(tag, step=runner.iter)

        def after_train_iter(self,
                             runner,
                             batch_idx: int,
                             data_batch: DATA_BATCH = None,
                             outputs: Optional[dict] = None) -> None:
            """Record logs after training iteration.

            Args:
                runner (Runner): The runner of the training process.
                batch_idx (int): The index of the current batch in the train loop.
                data_batch (dict tuple or list, optional): Data from dataloader.
                outputs (dict, optional): Outputs from model.
            """
            # Print experiment name every n iterations.
            if self.every_n_train_iters(
                    runner, self.interval_exp_name) or (self.end_of_epoch(
                runner.train_dataloader, batch_idx)):
                exp_info = f'Exp name: {runner.experiment_name}'
                runner.logger.info(exp_info)
            if self.every_n_inner_iters(batch_idx, self.interval):
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
                  and not self.ignore_last):
                # `runner.max_iters` may not be divisible by `self.interval`. if
                # `self.ignore_last==True`, the log of remaining iterations will
                # be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            runner.logger.info(log_str)
            runner.visualizer.add_scalars(
                tag, step=runner.iter + 1, file_path=self.json_log_path)
            self.log_metrics(tag, step=runner.iter + 1)


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
