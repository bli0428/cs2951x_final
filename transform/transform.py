import nltk

def transform(t):
    label = t.label()
    # switch statement based on t.label
    if label == 'Policy':
        return transform_Policy(t)

def transform_Policy(t):
    out_tree = nltk.Tree('Policy',[])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
            out_tree.append('entails that you')
        #ignores functors like :, \n, \t
        if label == 'ConditionalExecute':
            out_tree.append(transform_ConditionalExecute(elt))
    return out_tree

def transform_ConditionalExecute(t):
    out_tree = nltk.Tree('ConditionalExecute', [])
    for elt in t:
        label = elt.label()
        if label == 'If':
            out_tree.append(transform_If(elt))
        if label == 'Elif':
            out_tree.append(transform_Elif(elt))
        if label == 'Else':
            out_tree.append(transform_Else(elt))
    return out_tree

#note: current structure of if statement parse clashes with this because 'Execute' is included as string literal.
def transform_If(t):
    out_tree = nltk.Tree('If', ['if'])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolExp':
            out_tree.append(transform_BoolExp(elt))
        if label == 'Execute':
            out_tree.insert(0, transform_Execute(elt))
    return out_tree

def transform_Elif(t):
    out_tree = nltk.Tree('Elif', ['otherwise, if'])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolExp':
            out_tree.append(transform_BoolExp(elt))
        if label == 'Execute':
            out_tree.append(transform_Execute(elt))
    return out_tree

def transform_Else(t):
    out_tree = nltk.Tree('Else', ['and', transform_Execute(t[1]), 'otherwise'])
    return out_tree

def transform_Execute(t):
    out_tree = nltk.Tree('Execute', ['do', t[1]])
    return out_tree

#should include conjs too, but recursion errors means I put that off.
def transform_BoolExp(t):
    out_tree = nltk.Tree('BoolExp', [])
    for elt in t:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolTest':
            out_tree.append(transform_BoolTest(elt))
        if label == 'Value':
            out_tree.append(transform_Value(elt))
    return out_tree

def transform_BoolTest(t):
    out_tree = nltk.Tree('BoolTest',[])
    elt = t[0]
    if elt == '==':
        out_tree.append('is equal to')
    if elt == '!=':
        out_tree.append('is not equal to')
    if elt == '>':
        out_tree.append('is greater than')
    if elt == '<':
        out_tree.append('is less than')
    return out_tree

def transform_Value(t):
    pass

def transform_VariableName(t):
    return t[0]

def transform_Conj(t):
    pass

#Newline, Colon, Tab are cut out for the English translation