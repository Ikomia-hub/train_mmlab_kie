# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
import os
from ikomia.utils import strtobool
from ikomia.core import config as ikcfg

# Your imports below
from pathlib import Path
from datetime import datetime
from train_mmlab_kie.utils import prepare_dataset, register_mmlab_modules
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmocr.utils import register_all_modules
import logging
import shutil
from typing import Union, Dict
import yaml

ConfigType = Union[Dict, Config, ConfigDict]


class MyRunner(Runner):

    @classmethod
    def from_custom_cfg(cls, cfg, custom_hooks, visualizer) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=custom_hooks,
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=visualizer,
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainMmlabKieParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "sdmgr"
        self.cfg["cfg"] = "sdmgr_novisual_60e_wildreceipt.py"
        self.cfg["model_weight_file"] = ""
        self.cfg["config_file"] = ""
        self.cfg["pretrain"] = True
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 32
        self.cfg["dataset_split_ratio"] = 90
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["eval_period"] = 1
        self.cfg["dataset_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/dataset"
        self.cfg["expert_mode"] = False

    def set_values(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["cfg"] = param_map["cfg"]
        self.cfg["config_file"] = param_map["config_file"]
        self.cfg["model_weight_file"] = param_map["model_weight_file"]
        self.cfg["pretrain"] = strtobool(param_map["pretrain"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_ratio"] = int(param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["expert_mode"] = strtobool(param_map["expert_mode"])


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainMmlabKie(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        register_mmlab_modules()

        # Variable to check if the training must be stopped by user
        self.stop_train = False
        self.out_folder = ""
        if param is None:
            self.set_param_object(TrainMmlabKieParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        self.stop_train = False

        # Get param
        param = self.get_param_object()

        # Get input dataset
        input = self.get_input(0)

        # Current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            return

        # Output directory
        self.out_folder = Path(param.cfg["output_folder"] + "/" + str_datetime)
        self.out_folder.mkdir(parents=True, exist_ok=True)

        # Tensorboard
        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)

        # Transform Ikomia dataset to ICDAR compatible dataset if needed
        prepare_dataset(input.data, param.cfg["dataset_split_ratio"] / 100, param.cfg["dataset_folder"])

        # Create config from config file and parameters
        if not param.cfg["expert_mode"]:
            config, ckpt = self.get_cfg_and_weights_from_name(param)
            cfg = Config.fromfile(config)

            if "class_list" not in input.data["metadata"]:
                raise Exception("Dataset metadata should contain a key class_list. ")

            if "dict_file" not in input.data["metadata"]:
                raise Exception("Dataset metadata should contain a key dict_file")

            with open(input.data["metadata"]["dict_file"], 'r') as f:
                num_classes = len(f.read().rstrip().splitlines())

            shutil.copy2(input.data["metadata"]["dict_file"], self.out_folder)
            shutil.copy2(input.data["metadata"]["class_list"], self.out_folder)

            cfg.model.dictionary = dict(
                type='Dictionary',
                dict_file=input.data["metadata"]["dict_file"],
                with_padding=True,
                with_unknown=True,
                unknown_token=None)

            cfg.work_dir = str(self.out_folder)
            eval_period = param.cfg["eval_period"]
            cfg.evaluation = dict(interval=eval_period, metric=["kie/micro_f1"],
                                  rule="greater")
            cfg.val_evaluator = dict(
                type='F1Metric',
                mode='micro',
                num_classes=num_classes,
                ignored_classes=input.data["metadata"]["eval_ignore"] if "eval_ignore" in input.data["metadata"]
                else [])
            cfg.test_evaluator = cfg.val_evaluator

            cfg.data_root = str(Path(param.cfg["dataset_folder"]))
            data_type = "WildReceiptDataset"
            train = dict(
                metainfo=input.data["metadata"]['class_list'],
                type=data_type,
                ann_file=str(Path(cfg.data_root) / 'train.txt'),
                pipeline=cfg.train_pipeline)
            test = dict(
                metainfo=input.data["metadata"]['class_list'],
                type=data_type,
                ann_file=str(Path(cfg.data_root) / 'test.txt'),
                pipeline=cfg.test_pipeline,
                test_mode=True
            )

            cfg.train_dataloader.dataset = train
            cfg.test_dataloader.dataset = test
            cfg.val_dataloader.dataset = test

            cfg.train_dataloader.batch_size = param.cfg["batch_size"]
            cfg.train_dataloader.num_workers = 0
            cfg.train_dataloader.persistent_workers = False

            cfg.test_dataloader.batch_size = param.cfg["batch_size"]
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False

            cfg.val_dataloader.batch_size = param.cfg["batch_size"]
            cfg.val_dataloader.num_workers = 0
            cfg.val_dataloader.persistent_workers = False

            cfg.load_from = ckpt if param.cfg["pretrain"] else None

            cfg.train_cfg.max_epochs = param.cfg["epochs"]
            cfg.train_cfg.val_interval = eval_period

        else:
            config = param.cfg["config_file"]
            cfg = Config.fromfile(config)

        amp = True
        # save only best and last checkpoint
        cfg.checkpoint_config = None
        if "checkpoint" in cfg.default_hooks:
            cfg.default_hooks.checkpoint["interval"] = -1
            cfg.default_hooks.checkpoint["save_best"] = 'kie/micro_f1'
            cfg.default_hooks.checkpoint["rule"] = 'greater'

        cfg.visualizer.vis_backends = [dict(type='TensorboardVisBackend', save_dir=tb_logdir)]

        try:
            visualizer = Visualizer.get_current_instance()
        except:
            visualizer = cfg.get('visualizer')

        # register all modules in mmdet into the registries
        # do not init the default scope here because it will be init in the runner
        register_all_modules(init_default_scope=False)

        # enable automatic-mixed-precision training
        if amp:
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_log(
                    'AMP training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', (
                    '`--amp` is only supported when the optimizer wrapper type is '
                    f'`OptimWrapper` but got {optim_wrapper}.')
                cfg.optim_wrapper.type = 'AmpOptimWrapper'
                cfg.optim_wrapper.loss_scale = 'dynamic'

        custom_hooks = [
            dict(type='CustomHook', stop=self.get_stop, output_folder=str(self.out_folder),
                 emit_step_progress=self.emit_step_progress, priority='LOWEST'),
            dict(type='CustomMlflowLoggerHook', log_metrics=self.log_metrics)
        ]

        # build the runner from config
        runner = MyRunner.from_custom_cfg(cfg, custom_hooks, visualizer)

        runner.cfg = cfg
        # start training
        runner.train()

        print("Training finished!")
        # Call end_task_run to finalize process
        self.end_task_run()

    @staticmethod
    def get_model_zoo():
        configs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "kie")
        available_pairs = []
        for model_name in os.listdir(configs_folder):
            if model_name.startswith('_'):
                continue
            yaml_file = os.path.join(configs_folder, model_name, "metafile.yml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)
                    if 'Models' in models_list:
                        models_list = models_list['Models']
                    if not isinstance(models_list, list):
                        continue
                for model_dict in models_list:
                    available_pairs.append({"model_name": model_name, "cfg": os.path.basename(model_dict["Name"])})
        return available_pairs

    @staticmethod
    def get_cfg_and_weights_from_name(param):
        model_name = param.cfg["model_name"]
        model_config = param.cfg["cfg"]
        yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "kie", model_name,
                                 "metafile.yml")

        if model_config.endswith('.py'):
            model_config = model_config[:-3]
        if os.path.isfile(yaml_file):
            with open(yaml_file, "r") as f:
                models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

            available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"],
                                                       'ckpt': model_dict["Weights"]}
                                  for model_dict in models_list}
            if model_config in available_cfg_ckpt:
                cfg_file = available_cfg_ckpt[model_config]['cfg']
                ckpt_file = available_cfg_ckpt[model_config]['ckpt'] if param.cfg["model_weight_file"] == "" else param.cfg["model_weight_file"]
                cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_file)
            else:
                raise Exception(
                    f"{model_config} does not exist for {model_name}. Available configs for are {', '.join(list(available_cfg_ckpt.keys()))}")
        else:
            raise Exception(f"Model name {model_name} does not exist.")
        return cfg_file, ckpt_file

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainMmlabKieFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_mmlab_kie"
        self.info.short_description = "Train for MMOCR from MMLAB KIE models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "2.0.1"
        self.info.max_python_version = "3.10.0"
        self.info.icon_path = "icons/mmlab.png"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.original_repository = "https://github.com/open-mmlab/mmocr"

        self.info.repository = "https://github.com/Ikomia-hub/train_mmlab_kie"
        # Keywords used for search
        self.info.keywords = "train, key, information, extraction, kie, mmlab, sdmgr"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OCR"

    def create(self, param=None):
        # Create process object
        return TrainMmlabKie(self.info.name, param)
