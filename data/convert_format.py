import json 
import os 

path = os.path.dirname(os.path.abspath(__file__))

def convert(filename):
    with open(os.path.join(path, filename+'.json'), 'r', encoding='utf8') as f:
        data = json.loads(f.read())
        
    # print(data['data'][:1])

    hugging_dataset = []
    for ent in data['data']:
        title = ent['title']
        for paragraph in ent['paragraphs']:
            context = paragraph['context']
            for qas in paragraph['qas']:
                qid = qas['id']
                question = qas['question']
                text = []
                answer_start = []
                for ans in qas['answers']:
                    text.append(ans['text'])
                    answer_start.append(ans['answer_start'])
                answers = {
                    'text': text,
                    'answer_start': answer_start
                }
                ins = {'id': qid,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers}
                hugging_dataset.append(ins) 
    print(hugging_dataset[:1])
        
    
    with open(os.path.join(path, filename+'-f1.json'), 'w', encoding='utf8') as f:
        f.write(
            '\n'.join(json.dumps(i, ensure_ascii=False) for i in hugging_dataset)) 


if __name__ == '__main__':
    filename = "qa_dev_data"
    convert(filename=filename)