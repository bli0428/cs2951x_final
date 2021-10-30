from nltk import CFG
from nltk.parse.generate import generate
from nltk.grammar import Production, Nonterminal
from nltk.parse import ShiftReduceParser

def main():
    grammar = CFG.fromstring(open("rlang_subset.cfg", 'r').read())
    productions = grammar.productions()

    sr = ShiftReduceParser(grammar)
    sentence2 = 'Execute a'.split()
    print(sentence2)
    print(sr.parse(sentence2))
    
    for t in sr.parse(sentence2):
        print(t)

    extra_productions = list()
    # NLTK does not like some characters
    extra_productions.append(Production(Nonterminal('NewLine'), ['\n']))
    extra_productions.append(Production(Nonterminal('BoolTest'), ['==']))
    extra_productions.append(Production(Nonterminal('BoolTest'), ['!=']))
    extra_productions.append(Production(Nonterminal('BoolTest'), ['<']))
    extra_productions.append(Production(Nonterminal('BoolTest'), ['>']))
    extra_productions.append(Production(Nonterminal('Operation'), ['+']))
    extra_productions.append(Production(Nonterminal('Operation'), ['-']))
    extra_productions.append(Production(Nonterminal('Operation'), ['*']))
    extra_productions.append(Production(Nonterminal('Operation'), ['/']))
    extra_productions.append(Production(Nonterminal('Assignment'), [':=']))
    extra_productions.append(Production(Nonterminal('Colon'), [':']))
    extra_productions.append(Production(Nonterminal('Tab'), ['\t']))
    # You can also expand the number of possible variable names and values here
    extra_productions.append(Production(Nonterminal('VariableName'), ['alfred']))
    extra_productions.append(Production(Nonterminal('VariableName'), ['primitive_elements']))

    productions.extend(extra_productions)

    # print(productions)
    grammar = CFG(Nonterminal('Program'), productions)
    # print(grammar)
    
    
    # for sentence in generate(grammar, start=Nonterminal('Program'), depth=6):
    #     print(' '.join(sentence))


if __name__ == '__main__':
    main()
