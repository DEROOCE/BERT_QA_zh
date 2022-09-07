import logging 
import os 
import random
import numpy as np
import datetime as dt

from config import ModelConfig

import torch 
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import (BertConfig, BertTokenizer, 
                        BertForQuestionAnswering, XLMConfig, 
                        XLMForQuestionAnswering, XLMTokenizer, 
                        XLNetConfig, XLNetTokenizer,
                        XLNetForQuestionAnswering)
from transformers import get_linear_schedule_with_warmup
import mlflow 

from utils_squad1 import (read_squad_examples, convert_examples_to_features,
                        RawResult, 
                        RawResultExtended, 
                        write_predictions, write_predictions_extended)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train', 
        args.model_name_or_path.split(os.path.sep)[-1], # list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


class BERTQA(pl.LightningModule):
    def __init__(self, config, model, tokenizer, t_total):
        super().__init__()
        self.save_hyperparameters(ignore=['model'], logger=True)
        self.automatic_optimization = False
        self._global_step = 0
        self._tr_loss = 0.0
        self._logging_loss = 0.0
        self.model = model
        self.model.zero_grad()
        self.config = config 
        self.tokenizer = tokenizer
        self.test_results = []
        # training parameters
        self.t_total = t_total

    def forward(self, inputs):
        return self.model(**inputs)
    
    def test_dataloader(self):
        dataset, examples, features = load_and_cache_examples(self.config, 
                                                            self.tokenizer, 
                                                            evaluate=True, 
                                                            output_examples=True)
        self.config.eval_batch_size = self.config.per_gpu_eval_batch_size * max(1, self.config.n_gpu)
        test_iter = DataLoader(dataset, shuffle=False, 
                                    batch_size=self.config.eval_batch_size,
                                    num_workers=4,
                                    drop_last=False)
        logging.info("The number of test batchs is %d", len(test_iter))
        self.examples = examples 
        self.features = features
        
        return test_iter

    def training_step(self, batch, batch_idx):
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", self.config.num_train_epochs)
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)

        
        batch = tuple(t for t in batch)
        inputs = {'input_ids':     batch[0],
                'attention_mask':  batch[1], 
                'token_type_ids':  None if self.config.model_type == 'xlm' else batch[2],  
                'start_positions': batch[3], 
                'end_positions':   batch[4]}
        
        outputs = self(inputs)
        loss = outputs[0]
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        mlflow.log_metric("loss", loss)
        
        self._tr_loss += loss.item()
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            logging.info("The gradient is accumulated, then update the parameters, the batch index is %d, the global step is %d", batch_idx, self._global_step)
            self.optimizers().step()
            self.lr_schedulers().step()
            #self.lr_scheduler.step()
            self.model.zero_grad()
            self._global_step += 1
            self._tr_loss = 0 # reset the state 
                    
        mlflow.log_metric("training loss", self._tr_loss)
        self.log("loss", loss,
                on_step=True, prog_bar=True, logger=True)
        self.log("training loss", self._tr_loss,
                on_epoch=True, prog_bar=True, logger=True)
        logger.info(" global_step = %s, average loss = %s", self._global_step, self._tr_loss)
        
        return {"loss": loss, 'tr_loss': self._tr_loss}
    
    def test_step(self, batch, batch_idx):
        # Eval!
        logger.info("***** Running evaluation: batch_idx = %d *****", batch_idx)
        logger.info("  Batch size = %d", self.config.eval_batch_size) 
        
        batch = tuple(t for t in batch)
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': None if self.config.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                    }
        example_indices = batch[3]
        if self.config.model_type in ['xlnet', 'xlm']:
            inputs.update({'cls_index': batch[4],
                            'p_mask':    batch[5]})       
        outputs = self(inputs)
        
        for i, example_index in enumerate(example_indices):
            eval_feature = self.features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if self.config.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                        start_top_log_probs  = to_list(outputs[0][i]),
                                        start_top_index      = to_list(outputs[1][i]),
                                        end_top_log_probs    = to_list(outputs[2][i]),
                                        end_top_index        = to_list(outputs[3][i]),
                                        cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            self.test_results.append(result)   
        
        #return result

    def test_epoch_end(self, outputs):
        # Compute predictions
        output_prediction_file = os.path.join(self.config.output_dir, "predictions.json")
        output_nbest_file = os.path.join(self.config.output_dir, "nbest_predictions.json")
        if self.config.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.config.output_dir, "null_odds.json")
        else:
            output_null_log_odds_file = None

        logging.info("******Starting to write predictions*******")
        if self.config.model_type in ['xlnet', 'xlm']:
            # XLNet uses a more complex post-processing procedure
            write_predictions_extended(self.examples, self.features, self.test_results, 
                                    self.config.n_best_size,
                                    self.config.max_answer_length, output_prediction_file,
                                    output_nbest_file, output_null_log_odds_file, self.config.predict_file,
                                    self.model.config.start_n_top, self.model.config.end_n_top,
                                    self.config.version_2_with_negative, self.tokenizer, self.config.verbose_logging)
        else:
            write_predictions(self.examples, self.features, self.test_results, self.config.n_best_size,
                            self.config.max_answer_length, self.config.do_lower_case, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file, self.config.verbose_logging,
                            self.config.version_2_with_negative, self.config.null_score_diff_threshold)
        logging.info("******End of writing predictions*******")
        
        # Evaluate with the official SQuAD script
        evaluate_options = EVAL_OPTS(data_file=self.config.predict_file,
                                    pred_file=output_prediction_file,
                                    na_prob_file=output_null_log_odds_file)
        metrics = evaluate_on_squad(evaluate_options)
        print("The test metrcis are ", metrics)
        
        return metrics

        
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=self.config.learning_rate, 
                        eps=self.config.adam_epsilon)
        scheduler = scheduler = get_linear_schedule_with_warmup(optimizer, 
                            num_warmup_steps=self.config.warmup_steps, 
                            num_training_steps=self.t_total)
        # set the update step for the scheduler
        scheduler = {
        'scheduler': scheduler,
        'interval': 'step', # or 'epoch'
        'frequency': 1
        }
        return [optimizer], [scheduler]
    

def main():
    args = ModelConfig()
    do_train = False 
    do_test = True 

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. \
                        Use --overwrite_output_dir to overcome.".format(args.output_dir))    
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%a %d %b %Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=os.path.join(args.log_dir, "pl_training.log"),
                        filemode='w')

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, 
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    logger.info("Training/evaluation parameters %s", args)

    if do_train:
        args.n_gpu = 1
        args.per_gpu_train_batch_size = 8
        args.gradient_accumulation_steps = 32
        # Set seed
        set_seed(args)    
        # Set seed
        set_seed(args)    
        cur_time = dt.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
        
        ckpt_callback_loss = ModelCheckpoint(
                    monitor="loss", dirpath=args.cache_dir, 
                    filename=f'bert_QA_{cur_time}',
                    mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        

        trainer = pl.Trainer(max_epochs=1, 
                            callbacks=[ckpt_callback_loss, lr_monitor],
                            accelerator="gpu", devices=[7],
                            log_every_n_steps=50,
                            enable_progress_bar=True
                            )
        # train_loader
        train_dataset = load_and_cache_examples(args, 
                                                tokenizer, 
                                                evaluate=False, output_examples=False)
        logger.info("  Num examples = %d", len(train_dataset))
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_iter = DataLoader(train_dataset, shuffle=True, 
                                batch_size=args.train_batch_size,
                                num_workers=4,
                                drop_last=True)
        
        logging.info("The number of training batchs is %d", len(train_iter))
        # gradient_accumulation_steps 
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_iter) // args.gradient_accumulation_steps) + 1
            logging.info("Total optimization steps = %d", t_total)
        else:
            t_total = len(train_iter) // args.gradient_accumulation_steps * args.num_train_epochs
            logging.info("Total optimization steps = %d", t_total)

        
        
        mlflow.set_tracking_uri("")  # mlflow monitoring web url here
        mlflow.set_experiment("BERT_QA_Lightning")
        mlflow.start_run(run_name="%s_BERT_QA" 
                            % cur_time)
        bert_qa = BERTQA(args, model, tokenizer, t_total)
        # mlflow.pytorch.autolog()
        trainer.fit(bert_qa, train_dataloaders=train_iter)
        mlflow.end_run()
        print("training has been ended")

    if do_test:
        trainer = pl.Trainer(max_epochs=1, 
                            accelerator="gpu", devices=[7],
                            log_every_n_steps=50,
                            enable_progress_bar=True
                            )
        # load the fine_tuned model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache') 
        ckpt_name = os.listdir(model_path)[-1]
        logging.info("Using the fine-tuned model %s", ckpt_name)
        print(f"Using the model {ckpt_name}")
        load_path = os.path.join(model_path, ckpt_name)
        bert_ft = BERTQA.load_from_checkpoint(load_path, model=model)
        trainer.test(bert_ft) # without fine-tuning
        print("test ended")
    
if __name__ == "__main__":
    main()
