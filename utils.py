import os.path
import random
import numpy as np
import json
from mmcv.runner.hooks import HOOKS, Hook

class UserStop(Exception):
    pass

# Define custom hook to stop process when user uses stop button and to save last checkpoint
try:
    @HOOKS.register_module()
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
                runner.save_checkpoint(self.output_folder,"latest.pth",create_symlink=False)
                raise UserStop
except:
    pass

kie_models = {
            'SDMGR': {
                'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }

def prepare_dataset(ikdataset, split, output_dataset):
    n_img = len(ikdataset['images'])
    n_train = int(split*n_img)
    # Indices of train images
    idx_train = random.sample(list(np.arange(n_img)),n_train)
    train_text_file = os.path.join(output_dataset,"train.txt")
    test_text_file = os.path.join(output_dataset,"test.txt")
    rewrite = True
    if not(os.path.isdir(output_dataset)):
        os.mkdir(output_dataset)
    if os.path.isfile(train_text_file):
        # check the number of lines in the train.txt to know if dataset must be rewrote
        with open(train_text_file,'r') as f :
            rewrite = n_train != len(f.readlines())
    # If train.txt has not as much lines as intended, the function rewrites train.txt and test.txt
    # to fit the split ratio
    if rewrite:
        # Else
        print("Preparing dataset...")
        with open(train_text_file,'w') as f:
            f.write('')
        with open(test_text_file, 'w') as f:
            f.write('')

        for i,img in enumerate(ikdataset['images']):
            if i in idx_train:
                file_to_write = train_text_file
            else:
                file_to_write = test_text_file
            record={}
            record["file_name"] = img["filename"]
            record["height"] = img["height"]
            record["width"] = img["width"]
            record["annotations"]=[]
            for annot in img["annotations"]:
                sample={}
                sample['label'] = annot["category_id"]
                sample['text'] = annot['text']
                sample['box'] = annot['segmentation_poly'][0]
                record["annotations"].append(sample)
            dict_to_write = json.dumps(record)
            with open(file_to_write,'a') as f:
                f.write(dict_to_write+'\n')
        print("Dataset prepared!")
