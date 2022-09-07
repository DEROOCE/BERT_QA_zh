import os 
import torch 

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        # self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.predict_fire_name = "cmrc2018_dev_copy"
        self.train_file = os.path.join(self.dataset_dir, 'cmrc2018_train.json')
        self.predict_file = os.path.join(self.dataset_dir, 'cmrc2018_dev.json')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, 'model.pt')
        self.model_type = "bert"
        self.model_name_or_path =  os.path.join(self.project_dir, 'model', 'bert-base-chinese')
        self.config_name = None
        self.tokenizer_name = None # Pretrained tokenizer name or path if not the same as model_name"
        self.do_train = False
        self.do_eval = False
        self.evaluate_during_training = False
        self.do_lower_case = False # If null_score - best_non_null is greater than the threshold predict null.
        self.per_gpu_train_batch_size = 8 # 8
        self.per_gpu_eval_batch_size = 8
        self.learning_rate = 3e-5
        self.num_train_epochs = 5
        self.max_steps = -1 # If null_score - best_non_null is greater than the threshold predict null.
        self.max_seq_length = 384
        self.max_query_length = 64
        self.max_answer_length = 30 
        self.doc_stride = 128
        self.output_dir = os.path.join(self.project_dir, "output")
        self.cache_dir = os.path.join(self.project_dir, "cache")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.version_2_with_negative = False # If true, the SQuAD examples contain some that do not have an answer.'
        self.null_score_diff_threshold = 0.0 # If null_score - best_non_null is greater than the threshold predict null.
        self.gradient_accumulation_steps = 32  # If null_score - best_non_null is greater than the threshold predict null.
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_steps = 0
        self.n_best_size = 20  # If null_score - best_non_null is greater than the threshold predict null.
        self.verbose_logging = True
        self.logging_steps = 10
        self.save_steps = 50
        self.eval_all_checkpoints = False
        self.overwrite_output_dir = True
        self.overwrite_cache = False
        self.seed = 42

        self.local_rank = -1
        # self.no_cuda = True
        #self.device = 'cpu'
        #self.n_gpu = 0

        self.fp16 = False
        self.fp16_opt_level = "O1"  
        self.server_ip = ""
        self.server_port = ""

if __name__ == "__main__":
    args = ModelConfig()
    print(list(filter(None, args.model_name_or_path.split('/'))).pop())
    print(args.model_name_or_path.split(os.path.sep)[-1])
