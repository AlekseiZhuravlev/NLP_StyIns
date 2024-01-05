
import json
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


source_path = '/home/s94zalek/StyIns/GYAFC_Corpus'
target_path = '/home/s94zalek/StyIns/corpus'
categories = ['Entertainment_Music', 'Family_Relationships']

def preprocess_gyafc(source_file):
    # read the file, for each line, segment each sentence into tokens 
    # and then convert each token into lowercase. use a special symbol __NUM to represent digits
    
    with open(source_file, 'r') as f:
        lines = f.readlines()
        
        
    # tokenize with nltk
    lines = [word_tokenize(line) for line in lines]
    
    # convert to lowercase
    lines = [[token.lower() for token in line] for line in lines]
    
    # if token contains digits, replace it with __NUM
    for line in lines:
        for i in range(len(line)):
            if any(char.isdigit() for char in line[i]):
                line[i] = '__NUM'
    
    # join tokens into a sentence
    lines = [' '.join(line) for line in lines]
    
    return lines


def save_paired(lines_informal, lines_formal, target_file):
    # save the paired data into target_file
    
    with open(target_file, 'w') as f:
        for i in range(len(lines_informal)):
            item = [
                {"sent": lines_formal[i], "label": 1},
                {"sent": lines_informal[i], "label": 0},
            ]
            json.dump(item, f)
            f.write('\n')
            

def save_unpaired(lines_informal, lines_formal, target_file):
    # save the unpaired data into target_file
    
    with open(target_file, 'w') as f:
        for i in range(len(lines_informal)):
            item = [
                {"sent": lines_formal[i], "label": 1},
            ]
            json.dump(item, f)
            f.write('\n')
            
            item = [
                {"sent": lines_informal[i], "label": 0},
            ]
            json.dump(item, f)
            f.write('\n')
            
            
def save_train():
    
    split = 'train'
    
    full_informal = []
    full_formal = []
    
    for category in categories:
        source_file_informal = os.path.join(source_path, category, split, 'informal')
        source_file_formal = os.path.join(source_path, category, split, 'formal')
        
        lines_informal = preprocess_gyafc(source_file_informal)
        lines_formal = preprocess_gyafc(source_file_formal)
        
        full_informal.extend(lines_informal)
        full_formal.extend(lines_formal)
        
    save_paired(
        full_informal,
        full_formal,
        os.path.join(
            target_path,
            f'full_gyafc_paired_{split}.json',
        ),
    )
    save_unpaired(
        full_informal,
        full_formal,
        os.path.join(
            target_path,
            f'full_gyafc_unpaired_{split}.json',
        ),
    )
    print(f'Finish processing {split} split')
    

def save_val():
    
    # full_informal = []
    # full_formal = []
    
    # for category in categories:
    #     source_file_informal = os.path.join(source_path, category, 'tune', 'informal')
    #     source_file_formal = os.path.join(source_path, category, 'tune', 'formal.ref0')
        
    #     lines_informal = preprocess_gyafc(source_file_informal)
    #     lines_formal = preprocess_gyafc(source_file_formal)
        
    #     full_informal.extend(lines_informal)
    #     full_formal.extend(lines_formal)
    
    data = {
        'formal': [],
        'formal.ref0': [],
        'formal.ref1': [],
        'formal.ref2': [],
        'formal.ref3': [],
        'informal': [],
        'informal.ref0': [],
        'informal.ref1': [],
        'informal.ref2': [],
        'informal.ref3': [],
    }
    
    # read the data
    for category in categories:
        for file_name in data.keys():
            source_file = os.path.join(source_path, category, 'tune', file_name)
            lines = preprocess_gyafc(source_file)
            data[file_name].extend(lines)  
            
    # save paired data; original, 4 references, then reverse direction
    with open(os.path.join(
            target_path,
            f'full_gyafc_paired_val.json',
        ), 'w') as f:
        
        for i in range(len(data['formal'])):
            item = [
                {"sent": data['formal'][i], "label": 1},
                {"sent": data['informal.ref0'][i], "label": 0},
                {"sent": data['informal.ref1'][i], "label": 0},
                {"sent": data['informal.ref2'][i], "label": 0},
                {"sent": data['informal.ref3'][i], "label": 0},
            ]
            json.dump(item, f)
            f.write('\n')  
            
        for i in range(len(data['informal'])):
            item = [
                {"sent": data['informal'][i], "label": 0},
                {"sent": data['formal.ref0'][i], "label": 1},
                {"sent": data['formal.ref1'][i], "label": 1},
                {"sent": data['formal.ref2'][i], "label": 1},
                {"sent": data['formal.ref3'][i], "label": 1},
            ]
            json.dump(item, f)
            f.write('\n')
            
            
    # save unpaired data; formal, informal-ref0, formal-ref0, informal
    with open(os.path.join(
            target_path,
            f'full_gyafc_unpaired_val.json',
        ), 'w') as f:
        
        for i in range(len(data['formal'])):
            item = [
                {"sent": data['formal'][i], "label": 1},
            ]
            json.dump(item, f)
            f.write('\n')  
            
            item = [
                {"sent": data['informal.ref0'][i], "label": 0},
            ]
            json.dump(item, f)
            f.write('\n')
            
        for i in range(len(data['informal'])):
            item = [
                {"sent": data['formal.ref0'][i], "label": 1},
            ]
            json.dump(item, f)
            f.write('\n')
            
            item = [
                {"sent": data['informal'][i], "label": 0},
            ]
            json.dump(item, f)
            f.write('\n')
        
    print(f'Finish processing val split')
    
    
def save_test():
    
    full_informal = []
    full_formal = []
    
    for category in categories:
        source_file_informal = os.path.join(source_path, category, 'test', 'informal')
        source_file_formal = os.path.join(source_path, category, 'test', 'formal')
        
        lines_informal = preprocess_gyafc(source_file_informal)
        lines_formal = preprocess_gyafc(source_file_formal)
        
        full_informal.extend(lines_informal)
        full_formal.extend(lines_formal)
        
    with open('/home/s94zalek/StyIns/inps/gyafc_informal.txt', 'w') as f:
        for i in range(len(full_informal)):
            f.write(full_informal[i] + '\n')
            
    with open('/home/s94zalek/StyIns/inps/gyafc_formal.txt', 'w') as f: 
        for i in range(len(full_formal)):
            f.write(full_formal[i] + '\n')  
             
    print(f'Finish processing test split')
    

if __name__ == '__main__':

    save_train()
    save_val()
    save_test()

        # exit(0)
