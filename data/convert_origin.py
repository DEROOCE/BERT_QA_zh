import json 
import os 

path = os.path.dirname(os.path.abspath(__file__))

def convert(filename):
    with open(os.path.join(path, filename+'.json'), 'r', encoding='utf8') as f:
        data = json.loads(f.read())
        
    #print(data)

    with open(os.path.join(path, filename+'-o1.json'), 'w', encoding='utf8') as f:
        json.dump(data, f,
                    sort_keys=False, 
                    indent=4, 
                    separators=(',', ':'),
                    ensure_ascii=False)

    filename = "dev-v1.1"
    convert(filename=filename)