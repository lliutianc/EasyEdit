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


def edit_IKE_Blip2OPT_VQA(debug=False):
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2_local.yaml')
    editor = MultimodalEditor.from_hparams(hparams)

    if debug:
        size = 100
    else:
        size = None
    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)
    metrics, _, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)



def train_MEND_Blip2OPT_VQA(debug=False):
    if debug: 
        size = 100
    else:
        size = None

    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2_local.yaml')
    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()   


def edit_MEND_Blip2OPT_VQA(debug=False):
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2_local.yaml')
    editor = MultimodalEditor.from_hparams(hparams)

    if debug: 
        size = 100
    else:
        size = None

    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)

    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)



def train_SERAC_Blip2OPT_VQA(debug=False):
    if debug: 
        size = 100
    else:
        size = None

    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2_local.yaml')

    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()


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
    
    # Generate_Embedding_for_IKE()
    # edit_IKE_Blip2OPT_VQA(debug=False)

    # train_MEND_Blip2OPT_VQA()
    # edit_MEND_Blip2OPT_VQA()
    train_SERAC_Blip2OPT_VQA()
    edit_SERAC_Blip2OPT_VQA()
