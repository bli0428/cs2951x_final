import os
import sys
from nltk import CFG
from nltk.parse import RecursiveDescentParser

script_dir = os.path.dirname(__file__)

def parse(rlang_primitive): 
    input_rlang_file = f'rlang_{rlang_primitive}_output.txt'
    output_rlang_tokenized_file = f'tokenized_{rlang_primitive}_output.txt'
    cfg_file = f'rlang_{rlang_primitive}.cfg'
    common_cfg_file = f'../generate/cfgs/common.cfg'
    common_path = os.path.join(script_dir, common_cfg_file)
    grammar = CFG.fromstring(open(os.path.join(script_dir, "../generate/cfgs/" + cfg_file), 'r').read() + open(common_path, 'r').read())
    parser = RecursiveDescentParser(grammar)

    with open(os.path.join(script_dir, "../data/" + input_rlang_file), 'r') as f_input:
        with open(os.path.join(script_dir, "../data/" + output_rlang_tokenized_file), 'w') as f_output:
            lines = f_input.readlines()
            total_lines = len(lines)
            print(f"Writing tokenized file for {rlang_primitive} to ", output_rlang_tokenized_file)
            print(f'Tokenizing {total_lines} total statements')
            print('This may take a while...\n...')
            
            for i in range(total_lines):
                if (i % 1000 == 0):
                    print(f'Finished parsing {i}/{total_lines} statements')
                
                sentence = lines[i].split()
                print(parser.parse(sentence))
                for t in parser.parse(sentence):
                    f_output.write(' '.join(str(t).split()) + '\n')
    
    print("Done!")

def main(argv):
    valid_rlang = set(('constant', 'policy', 'predicate'))
    if len(argv) != 2:
        print('Invalid number of arguments')
        print(f'Expected input: `python3 parse.py <{valid_rlang}>`')
        print(f'i.e. `python3 parse.py policy`')
        return
    elif argv[1] not in valid_rlang:
        print(f'Invalid argument "{argv[1]}"". parse.py only parses the following RLang statements:', valid_rlang)
        print(f'{argv[1]} is not a valid RLang statement')
        return

    parse(argv[1])

if __name__ == '__main__':
    main(sys.argv)