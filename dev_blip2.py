import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer

import logging
from utils import set_logging_level
from edit_multimodal import print_result


logger = set_logging_level(logging.FATAL)


def train_MEND_Blip2OPT_VQA(debug=True):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2_local.yaml')
    # print(hparams)
    # return 

    if debug: 
        size = 100
    else:
        size = None


    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)
    
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    return 
    trainer.run()   


def test_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2_local.yaml')
    print(hparams)
    return 

    editor = MultimodalEditor.from_hparams(hparams)

    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=100)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=100)

    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)








def test():
    import json

    train_json = json.load(open('data/vqa/vqa_train.json', "r"))
    test_json = json.load(open('data/vqa/vqa_eval.json', "r"))

    train_imgs = []
    for data in train_json:
        train_imgs.append(data["image"])
        print(data)
        exit(1)

    test_imgs = []
    for data in test_json:
        test_imgs.append(data["image"])
    print(len(test_imgs))

    print(len(set(train_imgs).intersection(set(test_imgs))))



def edit_MEND_Blip2_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2_local.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )
    print_result(metrics)


    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=10)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=10)

    # print(train_ds.config.device)
    # print(editor.hparams.device)

    editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    print_result(metrics)


def edit_SERAC_Blip2OPT_VQA(debug=False):    
    if debug: 
        size = 100
    else:
        size = None

    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2_local.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    
    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)

    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
 



if __name__ == "__main__":
    
    """
    Note: 
    `train_MEND_Blip2OPT_VQA()` (MENDMultimodalTrainer) will create a checkpoint of MEND, then 
    `test_MEND_Blip2OPT_VQA()` (MENDMultimodalEditor) will load this checkpoint (named `archive`)
    """

    # train_MEND_Blip2OPT_VQA()
    # test_MEND_Blip2OPT_VQA()
    edit_MEND_Blip2_VQA()
    # edit_SERAC_Blip2OPT_VQA(True)