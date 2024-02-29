import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
from edit_multimodal import print_result


import logging
from utils import set_logging_level

logger = set_logging_level(logging.FATAL)


def train_MEND_MiniGPT4_VQA(debug=False):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4_local.yaml')
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
    
    trainer.run() 


def edit_MEND_MiniGPT4_VQA(debug=False):

    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4_local.yaml')
    editor = MultimodalEditor.from_hparams(hparams)

    if debug: 
        size = 100
    else:
        size = None
    train_ds = VQADataset('data/vqa/vqa_train.json', config=hparams, size=size)
    eval_ds = VQADataset('data/vqa/vqa_eval.json', config=hparams, size=size)

    metrics, edited_model, _ = editor.edit_dataset(
        ds=train_ds,
        train_ds=None, # Only useful for IKE
        keep_original_weight=True        
    )
    
    print_result(metrics)



if __name__ == "__main__":

    """
    NOTE: In the original file, train/test differs in which dataset is used to train the model. 
    """
    # train_MEND_MiniGPT4_VQA()
    edit_MEND_MiniGPT4_VQA(True)