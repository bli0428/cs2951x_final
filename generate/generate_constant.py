from nltk import CFG
from nltk.parse.generate import generate
from nltk.grammar import Production, Nonterminal
from nltk.parse import RecursiveDescentParser
import os

script_dir = os.path.dirname(__file__)
DEPTH = 8

def main():
    grammar = CFG.fromstring(open(os.path.join(script_dir, "./cfgs/rlang_constant.cfg"), 'r').read())
    productions = grammar.productions()

    grammar = CFG(Nonterminal('Program'), productions)   
    output_file = 'rlang_constant_output.txt'

    print('Generating CONSTANT statements with depth:', DEPTH)
    print('This may take a while...\n...')
    with open(os.path.join(script_dir, "../data/" + output_file), 'w') as f:
        count = 0
        for sentence in generate(grammar, start=Nonterminal('Program'), depth=DEPTH):
            # max out at 100k statements (arbitrarily picked. just didn't want files too large)
            if count == 100000:
                break
            f.write(' '.join(sentence) + '\n')
            count += 1
    print('Done! Written to data/', output_file)

if __name__ == '__main__':
    main()
