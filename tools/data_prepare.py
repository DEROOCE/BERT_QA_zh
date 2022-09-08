import os
import json 
import re


class DataPrepare():
    def __init__(self, 
                file_path, 
                data_path,
                data_type, 
                saved_file_name):
        super().__init__()
        self.file_path = file_path
        self.data_path = data_path
        self.data_type = data_type # train or test
        self.saved_file_name = saved_file_name
        
    def read_data(self):
        data = []
        with open(self.file_path, 'r', encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
            print("The length of data is ", len(data))
        
        return data

    @staticmethod
    def serialize_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        return obj
    
    def extract_data(self):
        all_data = []
        data = self.read_data()
        question_types = [""] # input questions 
        for idx, paragraph in enumerate(data):
            context_id = f"{self.data_type}_{idx}"
            # the context starts at the 27-th word
            context = paragraph['src'][27:].replace("(", "（").replace(")", "）")
            tgt = paragraph['tgt']
            # unify some special characters
            tgt = tgt.translate(str.maketrans({
                                    "(": "（",
                                    ")": "）",
                                    "+": "\\+",
                                    "{": "\\{",
                                    "}": "\\}",
                                    "^": "\\^",
                                    "[": "\\[",
                                    "]": "\\]", 
                                    "?": "\\?",
                                    "*": "\\*",
                                    ".": "\\.",
                                    "\\": "/",
                                    })) 
            text = re.findall(r"^.+: (.+)$", tgt)[0]  # answer text
            text_type =  re.findall(r"[a-zA-Z_]+", tgt)[0]  # the type of answer text
            qas = []
            for q_id, q_type in enumerate(question_types):
                question = f"这句话的{q_type}在哪里？"
                qas_id = f"{context_id}_QUERY_{q_id}"
                if text_type == q_type:
                    # 有答案
                    answer_start = re.search(text, context).span()[0] # find where the answer text starts
                    answers = {"text", text, "answer_start", answer_start}
                    qas.append({
                        "question": question,
                        "id": qas_id,
                        "answers": answers,
                    })
                else:
                    # 没有答案
                    answers = []
                    qas.append({
                        "question": question,
                        "id": qas_id,
                        "answers": answers,
                    })

            paragraphs = [{
                "id": context_id,
                "context": context,
                "qas": qas
            }]
            all_data.append({
                "paragraphs": paragraphs,
                "id": context_id,
                "title": paragraph['src'][27:33]  # title of the context
            })

        new_data = {
            "version": "v2.0",
            "data": all_data,
        }
        
        return new_data
        
    def write_data(self):
        data = self.extract_data()
        saved_file_path = os.path.join(self.data_path, self.saved_file_name)
        with open(saved_file_path, 'w', encoding="utf-8") as f:
                json.dump(data, f, 
                        indent=2,
                        ensure_ascii=False, 
                        default=self.serialize_sets)




if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"The project directory is {project_dir}")
    data_path = os.path.join(project_dir, 'data')
    train_file = os.path.join(data_path, "train_data.json")
    dev_file = os.path.join(data_path, 'test_data.json')
    
    dp_train = DataPrepare(file_path=train_file,
                    data_path=data_path,
                    data_type="TRAIN",
                    saved_file_name="qa_train_data.json")
    dp_train.write_data()
    
    dp_dev = DataPrepare(file_path=dev_file,
                    data_path=data_path,
                    data_type="DEV",
                    saved_file_name="qa_dev_data.json")
    dp_dev.write_data()
    
    








