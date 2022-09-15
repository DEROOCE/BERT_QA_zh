import logging 
from tqdm.auto import tqdm
import collections
import os 
import json
import datetime as dt
from argparse import ArgumentParser
import numpy as np
import torch 
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import default_data_collator
from datasets import Dataset
from datasets import load_metric
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import (generate_dataset, prepare_train_features, 
                prepare_validation_features, postprocess_qa_predictions,
                generate_dataloader)

logger = logging.getLogger(__name__)


class BERT_QA(pl.LightningModule):
    def __init__(self, args, model, tokenizer, 
                train_data, val_data, pred_data):
        super().__init__()
        self.save_hyperparameters(ignore=['model'], logger=True)
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.pred_data = pred_data
        
    def forward(self, input):
        return self.model(**input)

    def calculate_loss(self, batch, batch_idx):
        model_output = self(batch)
        loss = model_output.loss 
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx)
        self.log("loss", loss,
                prog_bar=True) # on_step=True
        self.logger.log_metrics(metrics={'loss': loss})
        return {'loss': loss}       
        
    def validation_step(self, batch, batch_idx):
        val_loss = self.calculate_loss(batch, batch_idx)
        self.log("val_loss", val_loss,
                prog_bar=True)
        self.logger.log_metrics(metrics={'val_loss': val_loss})
        return {'val_loss': val_loss} 
        
    def predict_step(self, batch, batch_idx):
        logging.info("Starting prediction step")
        output = self(batch)
        start_logits = output['start_logits']
        end_logits = output['end_logits']
        return {'start_logits': start_logits, 'end_logits': end_logits}

    def predict_step_end(self, predict_step_outputs):
        return predict_step_outputs

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=self.args.learning_rate, 
                        eps=self.args.adam_epsilon)
        # scheduler = scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                     num_warmup_steps=self.config.warmup_steps, 
        #                     num_training_steps=self.t_total)
        # # set the update step for the scheduler
        # scheduler = {
        #     'scheduler': scheduler,
        #     'interval': 'step', # or 'epoch'
        #     'frequency': 1
        # }
        return [optimizer] #, [scheduler]
        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='mlflow_name', 
                        help="Name of the experiment")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--squad_v2", type=bool, default=False,
                    help="The type of dataset is squad_v2 or squad_v1")
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased",
                        help="The pretrained model name on hugging face")
    parser.add_argument("--per_gpu_batch_size", type=int, default=16,
                        help="The number of batch size")
    parser.add_argument("--max_length", type=int, default=384,
                        help="The max length of input features")
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="The stride of two splits")
    parser.add_argument("--pad_on_right", type=bool, default=True,
                        help="pad the context on the right side")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The number of learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="The epoch number of training")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="The decay rate of the weight")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=50)
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="The number of available gpus")
    parser.add_argument("--do_train", type=bool, default=False,
                        help="Do training or testing")
    parser.add_argument("--do_pred", type=bool, default=False)
    parser.add_argument("--train_data_name", type=str, default="train-v2.012.json")
    parser.add_argument("--val_data_name", type=str, default="dev-v2.012.json")
    
    parser = Trainer.add_argparse_args(parser)  # 将Trainer中的参数加入到parser中
    args = parser.parse_args()
    

    # data path configuration
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, 'data')
    save_path = os.path.join(root_path, 'checkpoint')
    log_path = os.path.join(save_path, 'logs')
    if not os.path.exists(save_path):
        logger.info(f"The directory {save_path} doesn't exist.")
        os.makedirs(save_path)
    if not os.path.exists(log_path):
        logger.info(f"The directory {log_path} doesn't exist.")
        os.makedirs(log_path)
    
    time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(log_path, f'training_{time}.log'))
    
    #datasets = load_dataset('squad_v2' if args.squad_v2 else 'squad')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)
    
    train_data_path = os.path.join(data_path, args.train_data_name)
    val_data_path = os.path.join(data_path, args.val_data_name)
    train_data = generate_dataset(train_data_path)
    val_data = generate_dataset(val_data_path)
    bert_qa = BERT_QA(args, model, tokenizer, train_data, val_data, pred_data=val_data)

    # configuration for MLFlowLogger
    os.environ['MLFLOW_S3_ENDPOINT_URL']=''
    mlf_logger = MLFlowLogger(experiment_name=args.exp_name, 
                            run_name=dt.datetime.now().strftime('%Y%m%d%H%M%S'),
                            tracking_uri='')
    
    mlf_logger.log_hyperparams({
                                'learning_rate': args.learning_rate ,
                                'batch_size': args.per_gpu_batch_size ,
                                'max_epochs': args.num_train_epochs,
                                'accumulate_grad_batches': args.accumulate_grad_batches ,
                                'limit_val_batches': args.limit_val_batches ,})

    if args.do_train:
        logging.info("Preparing the train dataloader and val dataloader")
        train_iter = generate_dataloader(train_data, prepare_train_features, args,
                                        tokenizer, default_data_collator,
                                        shuffle=True, drop_last=True)
        val_iter = generate_dataloader(val_data, prepare_train_features, args,
                                        tokenizer, default_data_collator)
        
        logging.info("End to prepare the train dataloader and val dataloader")
        cur_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ckpt_callback_loss = ModelCheckpoint(
                        monitor="loss", dirpath=os.path.join(save_path, cur_time),  # val_loss
                        filename='bert-QA-{loss:.2f}',
                        mode="min"
        )
        trainer = pl.Trainer(
                            accelerator='gpu', devices=[0], 
                            accumulate_grad_batches=args.accumulate_grad_batches,
                            #precision=args.precision, 
                            logger=mlf_logger,
                            callbacks=[ckpt_callback_loss],
                            #default_root_dir=save_path,
                            #logger=mlf_logger,
                            max_epochs=args.num_train_epochs,
                            #enable_checkpointing=False,
                            #gradient_clip_algorithm=args.gradient_clip_algorithm,
                            #gradient_clip_val=args.gradient_clip_val               #track_grad_norm=args.track_grad_norm,
                            val_check_interval=args.val_check_interval,
                            #check_val_every_n_epoch=1,	# 每n个train epoch执行一次验证
                            limit_val_batches=args.limit_val_batches,  # 30
                            #log_every_n_steps=50
                            )  
        trainer.fit(bert_qa, train_dataloaders=train_iter, val_dataloaders=val_iter)
        logging.info("training has been ended")
    
    if args.do_pred:
        logging.info("Preparing the predict dataloader")
        pred_iter = generate_dataloader(val_data, prepare_train_features, args,
                                tokenizer, default_data_collator)

        trainer = pl.Trainer(max_epochs=1, 
                            accelerator="gpu", devices=[0],
                            log_every_n_steps=50,
                            enable_progress_bar=True
                            ) 
        dir_name = ""   
        ckpt_name = ""  # name of the saved ckeckpoint file
        logging.info("Using the fine-tuned model %s", ckpt_name)
        print(f"Using the model {ckpt_name}")
        load_path = os.path.join(save_path, dir_name, ckpt_name)
        logging.info("Loading the pretrained model")
        bert_ft = BERT_QA.load_from_checkpoint(load_path, model=model)
        outputs = trainer.predict(bert_ft, dataloaders=pred_iter) # without fine-tuning    

        # format the output
        start_logits = []
        end_logits = []
        for output in outputs:
            start = output['start_logits']
            end = output['end_logits']
            start_logits.append(start)
            end_logits.append(end)
        start_logits = torch.vstack(start_logits)
        end_logits = torch.vstack(end_logits)
        print("start_logits.shape = ", start_logits.shape,
                " end_logits.shape = ", end_logits.shape)
        raw_predictions = [start_logits, end_logits]
        
        # format the original data
        features = val_data.map(
            prepare_validation_features,
            batched=True,
            remove_columns=val_data.column_names,
            fn_kwargs={"args": args, "tokenizer": tokenizer}
        )
        features.set_format(type=features.format["type"], 
                            columns=list(features.features.keys()))
        
        # postprocess for the output predictions
        final_predictions = postprocess_qa_predictions(args,
                                                    tokenizer,
                                                    val_data, 
                                                    features,
                                                    raw_predictions)   
        
        metric = load_metric("squad_v2" if args.squad_v2 else "squad") 
        if args.squad_v2:
            formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_data]
        results = metric.compute(predictions=formatted_predictions, references=references)
        print("results = ", results)        
        logging.info("testing has been ended")
    


