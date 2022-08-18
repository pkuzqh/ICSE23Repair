import pickle
from Searchnode1 import Node
def getroottree(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root
def getLocVar(node):
  varnames = []
  if node.name == 'VariableDeclarator':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'FormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'InferredFormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  for x in node.child:
    varnames.extend(getLocVar(x))
  return varnames
def getchildbyName(root, name):
  for x in root.child:
    if x.name == name:
      return x
  return None

import re
def is_number(num):
  pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
  result = pattern.match(num)
  if result:
    return True
  else:
    return False
def addTypeForIden(root, tdict):
  if len(root.child) == 0:
    if root.name in tdict:
      if tdict[root.name] in ['int', 'double', 'long', 'float']:
        root.type = 'numeric'
      elif tdict[root.name] in ['boolean']:
        root.type = 'bool'
      elif 'String' in tdict[root.name]:
        root.type = 'string'
      else:
        root.type = 'ptype'
    elif root.name == '<string>':
      root.type = 'string'
    elif root.name == 'null':
      root.type = 'null'
    elif root.father.name == 'operator':
      if root.name in ['&&', '||']:
        root.type = 'bool'
      elif root.name in ['==', '>=', '<=', '!=', '>', '<']:
        root.type = 'bool-numeric'
      else:
        root.type = 'numeric' 
    elif root.father.name == 'postfix_operators':
      root.type = 'numeric'
    elif root.father.name == 'prefix_operators':
      if root.name == '!':
        root.type = 'bool'
      else:
        root.type = 'numeric'
    elif root.father.name == 'value':
      if root.name in ['true', 'false']:
        root.type = 'bool'
      elif is_number(root.name):
        root.type = 'numeric'
      elif '0x' in root.name:
        root.type = 'numeric'
      else:
        print(root.name)
        root.type = 'numeric'
        #assert(0)
    else:
      root.type = 'init'
  for x in root.child:
    addTypeForIden(x, tdict)
def addTypeForNonTerm(root):
  for x in root.child:
    addTypeForNonTerm(x)
  if root.name == 'MemberReference':
    for x in root.child:
      if x.type in ['bool', 'numeric', 'string']:
        root.type = x.type
        break
      elif x.type == 'ptype' and root.type == 'init':
        root.type = 'ptype'
      else:
        root.type = 'init'
    pass
  elif root.name == 'operator' and len(root.child) > 0:
    root.type = root.child[0].type
  elif root.name in ['operandl', 'operandr']:
    root.type = root.child[0].type
  elif root.name == 'BinaryOperation':
    for x in root.child:
      if x.name == 'operator':
        if 'bool' in x.type:
          root.type = 'bool'
        else:
          root.type = x.type
  elif root.name == 'member':
    root.type = root.child[0].type
    assert(len(root.child) == 1)
  elif (root.name == 'value' and root.father and root.father.name == 'Literal') or (root.name == 'Literal' and len(root.child) > 0 and root.child[0].name == 'value'):
    root.type = root.child[0].type
    #print(root.printTreeWithType(root))
    assert(len(root.child) == 1)
  elif root.name in ['prefix_operators', 'postfix_operators']:
    for x in root.child:
      if x.type in ['bool', 'numeric']:
        root.type = x.type
        break
      else:
        pass
def turnold2new(tokens):
  if True:
        root = getroottree(tokens)
        varnames = getLocVar(root)
        vnum = 0
        fnum = 0
        vardic = {}
        typedic = {}
        for x in varnames:
            if x[1].name == 'VariableDeclarator':
                vnum += 1
                vardic[x[0]] = 'loc' + str(vnum)
                t = -1
                for s in x[1].father.father.child:
                    #print(s.name)
                    if s.name == 'type':
                      if len(s.child[0].child) == 1:#normal type
                        t = s.child[0].child[0].child[0].name
                        break
                      elif len(s.child[0].child) >= 2:#array type
                        t = s.child[0].child[0].child[0].name + '[]'
                        break
                      else:
                        assert(0)
                assert(t != -1)
                typedic[x[0]] = t
            else:
                fnum += 1
                vardic[x[0]] = 'par' + str(fnum)
                t = -1
                for s in x[1].child:
                    if s.name == 'type':
                      if len(s.child[0].child) == 1:
                        t = s.child[0].child[0].child[0].name
                        break
                      elif len(s.child[0].child) >= 2:
                        t = s.child[0].child[0].child[0].name + '[]'
                        break
                #print(root.printTree(x[1]))
                if t == -1:
                  continue
                typedic[x[0]] = t
        #print(typedic)
        addTypeForIden(root, typedic)
        addTypeForNonTerm(root)
  return root.printTreeWithType(root).split()
from tqdm import tqdm
import traceback
import sys
if __name__ == '__main__':
    v = int(sys.argv[1])
    #data = pickle.load('traindatawithtype%d.pkl'%v, 'rb')#pickle.load(open('/raid/zqh/copyhome/dedata.pkl', "rb"))
    data = pickle.load(open('dataextra.pkl', 'rb'))#pickle.load(open('/data/zqh/data/detection_dataset/data.pkl', "rb"))
    #data = pickle.load(open('/home/zqh/dedata.pkl', "rb"))
    data = data * 200
    #data.extend(pickle.load(open('data2.pkl', "rb")))
    data.extend(pickle.load(open('/raid/zqh/copyhome/data0.pkl', "rb")))
    data.extend(pickle.load(open('/raid/zqh/copyhome/data1.pkl', "rb")))
    print(len(data))
    data = data[10000 * v : 10000 * v +10000]
    newdata = []
    for entry in tqdm(data):
      try:
        if 'oldtree' in entry:
          tokens1 = entry['oldtree']
          tokens2 = entry['newtree']
        else:
          tokens1 = entry['old']
          tokens2 = entry['new']
        tokens = tokens1.split()
        newold = turnold2new(tokens)
        newfix = turnold2new(tokens2.split())
        newdata.append({'new':newfix, 'old':newold})#, 'id':entry['id'], 'df':entry['df']})
      except:
        traceback.print_exc()
        pass
    print(len(newdata))
    open('traindatawithtype%d.pkl'%v, 'wb').write(pickle.dumps(newdata))