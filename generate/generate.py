from os import path
import sys
from nltk import CFG
from nltk.parse.generate import generate
from nltk.grammar import Nonterminal

script_dir = path.dirname(__file__)
DEPTH = 8
EFFECT_DEPTH = 12
RLANG_TYPE_MAP = {
    'constant': 'Constant', 
    'policy': 'Policy',
    'action': 'Action',
    'option': 'Option', 
    'predicate': 'Predicate', 
    'markov_feature': 'MarkovFeature', 
    'effect': 'Effect'
}

"""
HOW TO USE:
    1. add rlang primitive/keyword you want to generate to `RLANG_TYPE_MAP` above
    2. create corresponding cfg at `./generate/cfgs/rlang_<primitive/keyword>.cfg`
    3. run python3 generate.py <primitive/keyword>
"""
def parse(rlang_type): 
    cfg_file = f'./cfgs/rlang_{rlang_type}.cfg'
    output_file = f'../data/rlang_{rlang_type}_output.txt'

    if not path.isfile(path.join(script_dir, cfg_file)):
        print(f"ERROR: {cfg_file} does not exist! Please create a CFG file for {rlang_type}")
        return

    grammar = CFG.fromstring(open(path.join(script_dir, cfg_file), 'r').read())
    productions = grammar.productions()

    grammar = CFG(Nonterminal(RLANG_TYPE_MAP[rlang_type]), productions)   

    # need to use larger depth for effects
    depth = EFFECT_DEPTH if rlang_type == 'effect' else DEPTH

    print(f'Generating {rlang_type.upper()} statements with depth:', depth)
    print('This may take a while...\n...')
    with open(path.join(script_dir, output_file), 'w') as f:
        count = 0
        for sentence in generate(grammar, start=Nonterminal(RLANG_TYPE_MAP[rlang_type]), depth=depth):
            # max out at 100k statements (arbitrarily picked. just didn't want files too large)
            if count == 100000:
                break
            f.write(' '.join(sentence) + '\n')
            count += 1

    print('Done! Written to', output_file)

def main(argv):
    valid_rlang = set(RLANG_TYPE_MAP.keys())
    if len(argv) != 2:
        print('Invalid number of arguments')
        print(f'Expected input: `python3 generate.py <{valid_rlang}>`')
        print(f'i.e. `python3 generate.py policy`')
        return
    elif argv[1] not in valid_rlang:
        print(f'Invalid argument "{argv[1]}"". generate.py only generate the following RLang statements:', valid_rlang)
        print(f'{argv[1]} is not a valid RLang statement')
        return

    parse(argv[1])

if __name__ == '__main__':
    main(sys.argv)