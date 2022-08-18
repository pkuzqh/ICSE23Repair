import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
import h5py
from tqdm import tqdm
import json
from scipy import sparse
from parse_dfg import *
sys.setrecursionlimit(500000000)
class Graph:
    def __init__(self):
        self.row = []
        self.col = []
        self.val = []
        self.edge = {}
        self.rowNum = 0
        self.colNum = 0
    def addEdge(self, r, c, v):
        if (r, c) in self.edge:
            print(r, c)
            assert(0)
        self.edge[(r, c)] = len(self.row)
        self.row.append(r)
        self.col.append(c)
        self.val.append(v)
        '''self.edge[(c, r)] = len(self.row)
        self.row.append(c)
        self.col.append(r)
        self.val.append(v)'''
    def editVal(self, r, c, v):
        self.val[self.edge(r, c)] = v
    def updateval(self, index, v):
        self.val[index] = v
    def normlize(self):
        r = {}
        c = {}
        for i  in range(len(self.row)):
            if self.row[i] not in r:
                r[self.row[i]] = 0
            r[self.row[i]] += 1
            if self.col[i] not in c:
                c[self.col[i]] = 0
            c[self.col[i]] += 1
        for i in range(len(self.row)):
            self.val[i] = 1 / math.sqrt(r[self.row[i]]) * 1 / math.sqrt(c[self.col[i]])
    def tonumpy(self):
        ans = np.zeros([self.rowNum, self.colNum])
        for i in range(len(self.row)):
            ans[self.row[i], self.col[i]] = self.val[i]
        return ans
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", isBeam=False):
        self.train_path = "train_process.txt"
        self.val_path = "dev_process.txt"  # "validD.txt"
        self.test_path = "test_process.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.Var_Len = config.varlen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50
        self.edgedic = {'pad':0}
        self.ruledict = pickle.load(open("train/data/rule.pkl", "rb"))
        assert('start -> copyword2' in self.ruledict)
        if self.dataName != 'train':
            self.edgedic = pickle.load(open('train/data/edge.pkl', 'rb'))
        #self.ruledict['start -> copyword2'] = len(self.ruledict)
        #print(self.ruledict)
        #self.ruledict["start -> Module"] = len(self.ruledict)
        #self.ruledict["start -> copyword"] = len(self.ruledict)
        self.rrdict = {}
        for x in self.ruledict:
            self.rrdict[self.ruledict[x]] = x
        if not os.path.exists("train/data/nl_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        #print(self.Nl_Voc)
        if dataName == "train":
            if os.path.exists("data.pkl"):
                self.data = pickle.load(open("data.pkl", "rb"))
                return
            data = pickle.load(open('process_datacopy.pkl', 'rb'))
            print(len(data))
            train_size = int(len(data) / 8 * 7)
            self.data = self.preProcessData(data)
        elif dataName == "val":
            if os.path.exists("valdata.pkl"):
                self.data = pickle.load(open("valdata.pkl", "rb"))
                self.nl = pickle.load(open("valnl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.val_path, "r", encoding='utf-8'))
        else:
            if os.path.exists("train/data/testdata.pkl"):
                #data = pickle.load(open('process_datacopy.pkl', 'rb'))
                #train_size = int(len(data) / 8 * 7)
                #data = data[train_size:]
                '''print(data[5])
                print(self.rrdict[152])
                print(data[6])
                print(data[11])
                print(data[13])
                exit(0)'''
                self.data = pickle.load(open("train/data/testdata.pkl", "rb"))
                if not isBeam:
                    tmp = []
                    for i in range(len(self.data)):
                        tmp2 = []
                        for j in range(len(self.data[i])):
                            if j in [1, 4, 8, 11, 29, 44, 58]:
                                continue
                            tmp2.append(self.data[i][j])
                        tmp.append(tmp2)
                    self.data = tmp
                #self.code = pickle.load(open("testcode.pkl", "rb"))
                self.nl = pickle.load(open("train/data/testnl.pkl", "rb"))
                #self.var = pickle.load(open('testvar.pkl', 'rb'))
                return
            data = pickle.load(open('train/data/testcopy.pkl', 'rb'))
            #train_size = int(len(data) / 8 * 7)
            self.data = self.preProcessData(data)
            #self.data = self.preProcessData(open(self.test_path, "r", encoding='utf-8'))

    def Load_Voc(self):
        if os.path.exists("train/data/nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("train/data/nl_voc.pkl", "rb"))
        if os.path.exists("train/data/code_voc.pkl"):
            self.Code_Voc = pickle.load(open("train/data/code_voc.pkl", "rb"))
        if os.path.exists("train/data/char_voc.pkl"):
            self.Char_Voc = pickle.load(open("train/data/char_voc.pkl", "rb"))
        self.Nl_Voc["<emptynode>"] = len(self.Nl_Voc)
        self.Code_Voc["<emptynode>"] = len(self.Code_Voc)

    def init_dic(self):
        print("initVoc")
        #f = open(self.train_path, "r", encoding='utf-8')
        #lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        nls = []
        rules = []
        data = pickle.load(open('process_datacopy.pkl', 'rb'))
        for x in data:
            if len(x['rule']) > self.Code_Len:
                continue
            nls.append(x['input'])
        '''for i in tqdm(range(int(len(lines) / 5))):
            data = lines[5 * i].strip().lower().split()
            nls.append(data)
            rulelist = lines[5 * i + 1].strip().split()
            tmp = []
            for x in rulelist:
                if int(x) >= 10000:
                    tmp.append(data[int(x) - 10000])
            rules.append(tmp)
        f.close()
        nl_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=0)
        code_voc = VocabEntry.from_corpus(rules, size=50000, freq_cutoff=10)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id'''
        code_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=10)
        self.Code_Voc = code_voc.word2id
        for x in self.ruledict:
            print(x)
            lst = x.strip().lower().split()
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.Code_Voc:
                    self.Code_Voc[y] = len(self.Code_Voc)
            #rules.append([lst[0]] + lst[2:])
        #print(self.Code_Voc)
        if 'numeric' not in self.Code_Voc:
            self.Code_Voc['numeric'] = len(self.Code_Voc)
        if 'string' not in self.Code_Voc:
            self.Code_Voc['string'] = len(self.Code_Voc)
        if 'bool' not in self.Code_Voc:
            self.Code_Voc['bool'] = len(self.Code_Voc)
        if 'ptype' not in self.Code_Voc:
            self.Code_Voc['ptype'] = len(self.Code_Voc)
        self.Code_Voc['[mask]'] = len(self.Code_Voc)
        self.Nl_Voc = self.Code_Voc
        #print(self.Code_Voc)
        assert("root" in self.Code_Voc)
        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
        print(maxNlLen, maxCodeLen, maxCharLen)
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def preProcessOne(self, data):
        #print(tree)
        #print(self.nl[0])
        inputNl = []
        inputNlchar = []
        inputPos = []
        inputNlad = []
        Nl = []
        inputVar = []
        inputUseable = []
        inputType = []
        classes = []
        methodname = []
        for xss in data:
            inputpos = xss['prob']
            tree = xss['tree']
            idss = xss['idss']
            rrdic = {}
            methodname.append('unknown')
            for x in xss['vardic']:
                if xss['vardic'][x] == 'meth0':
                    methodname[-1] = (x)
                    break
            if os.path.exists("generation/result/%s.json" % idss):
                classcontent = json.load(open("generation/result/%s.json" % idss, 'r'))
            else:
                classcontent = []
            rrdicts = {}
            classname = xss['classname']
            classes.append({'classes':[[]]})
            for x in classcontent:
                if x['filename'] == classname + '.java':
                    classes[-1] = x
                    break
                if 'package_name' in x and x['package_name'] + '.' + x['filename'] == classname + '.java':
                    classes[-1] = x
                    break
                
            inputpos = self.pad_seq(inputpos, self.Nl_Len)
            nl = tree.split()
            node = Node(nl[0], 0)
            currnode = node
            idx = 1
            #print(nl[-1])
            nltmp = [nl[0]]
            nodes = [node]
            for j, x in enumerate(nl[1:]):
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            Nl.append((nl, xss['varwithtype']))
            #print(nltmp)
            dfg, _ = dfg_java(node, {}, nodes)
            gcode = Graph()
            for edge in dfg:
                if edge[1] not in self.edgedic:
                    self.edgedic[edge[1]] = len(self.edgedic)
                for idx in edge[2]:
                    gcode.addEdge(edge[0], idx, self.edgedic[edge[1]])
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        if 'child->father' not in self.edgedic:
                            self.edgedic['child->father'] = len(self.edgedic) 
                        gcode.addEdge(x.id, x.father.id, self.edgedic['child->father'])
                        #nladrow.append(x.id)
                        #nladcol.append(x.father.id)
                        #nladdata.append(1)
                    idx = x.father.child.index(x)
                    if idx > 0:
                        if 'left->ad' not in self.edgedic:
                            self.edgedic['left->ad'] = len(self.edgedic)
                        gcode.addEdge(x.id, x.father.child[idx - 1].id, self.edgedic['left->ad'])
                    if idx < len(x.father.child) - 1:
                        if 'right->ad' not in self.edgedic:
                            self.edgedic['right->ad'] = len(self.edgedic)
                        gcode.addEdge(x.id, x.father.child[idx + 1].id, self.edgedic['right->ad'])
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        if 'father->child' not in self.edgedic:
                            self.edgedic['father->child'] = len(self.edgedic)
                        gcode.addEdge(x.id, s.id, self.edgedic['father->child'])
            nl = nltmp
            vars = xss['varwithtype']
            tmpvar = []
            tmptype = []
            tmpuseable = []
            for k, x in enumerate(vars):
                tmpvar.append(x[0])
                tmptype.append(x[1])
                tmpuseable.append(x[2] + 5) 
                for p, nodet in enumerate(nl):
                    if nodet == x[0]:
                        if 'comef' not in self.edgedic:
                            self.edgedic['comef'] = len(self.edgedic)
                        if 'usev' not in self.edgedic:
                            self.edgedic['usev'] = len(self.edgedic)
                        gcode.addEdge(p, self.Nl_Len + k, self.edgedic['comef'])
                        gcode.addEdge(self.Nl_Len + k, p, self.edgedic['usev'])
            if 'dtype' not in self.edgedic:
                self.edgedic['dtype'] = len(self.edgedic)
            if 'stype' not in self.edgedic:
                self.edgedic['stype'] = len(self.edgedic)
            for k, v in enumerate(tmpvar):
                for j, v in enumerate(tmpvar):
                    if tmptype[k] == tmptype[j]:
                        if tmptype[k] == 'ptype':
                            if vars[k][3] == vars[j][3]:
                                gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['stype'])
                            else:
                                gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['dtype'])
                        else:         
                            gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['stype'])
                    else:
                        gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['dtype'])
            inputVar.append(self.pad_seq(self.Get_Em(tmpvar, self.Nl_Voc), self.Var_Len))
            inputType.append(self.pad_seq(self.Get_Em(tmptype, self.Nl_Voc), self.Var_Len))
            inputUseable.append(self.pad_seq(tmpuseable, self.Var_Len))
            inputnls = self.pad_seq(self.Get_Em(nl, self.Nl_Voc), self.Nl_Len)
            nlad = sparse.coo_matrix((gcode.val, (gcode.row, gcode.col)), shape=(self.Nl_Len + self.Var_Len, self.Nl_Len + self.Var_Len))

            inputNl.append(inputnls)
            inputNlad.append(nlad)
            inputPos.append(inputpos)
        self.data = [inputNl, inputNlad, inputPos, inputVar, inputType, inputUseable]
        self.nl = Nl
        self.classes = classes
        self.method = methodname
        return
        #return np.array([inputnls]), np.array([nlad.toarray()]), np.array([inputpos]), np.array([inputnlchar])
    def preProcessData(self, dataFile):
        #lines = dataFile.readlines()
        inputNl = []
        inputNlad = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        inputPos = []
        inputVar = []
        inputType = []
        inputUseable = []
        nls = []
        vardics = []
        maxl = 0
        for i in tqdm(range(len(dataFile))):
            if len(dataFile[i]['rule']) > self.Code_Len:
                continue
            #vardics.append(dataFile[i]['vardic'])
            child = {}
            nl = dataFile[i]['input']#lines[5 * i].lower().strip().split()
            node = Node(nl[0], 0)
            currnode = node
            idx = 1
            nltmp = [nl[0]]
            nodes = [node]
            for x in nl[1:]:
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            dfg, _ = dfg_java(node, {}, nodes)
            gcode = Graph()
            for edge in dfg:
                if edge[1] not in self.edgedic:
                    self.edgedic[edge[1]] = len(self.edgedic)
                for idx in edge[2]:
                    gcode.addEdge(edge[0], idx, self.edgedic[edge[1]])
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        if 'child->father' not in self.edgedic:
                            self.edgedic['child->father'] = len(self.edgedic) 
                        gcode.addEdge(x.id, x.father.id, self.edgedic['child->father'])
                        #nladrow.append(x.id)
                        #nladcol.append(x.father.id)
                        #nladdata.append(1)
                    idx = x.father.child.index(x)
                    if idx > 0:
                        if 'left->ad' not in self.edgedic:
                            self.edgedic['left->ad'] = len(self.edgedic)
                        gcode.addEdge(x.id, x.father.child[idx - 1].id, self.edgedic['left->ad'])
                    if idx < len(x.father.child) - 1:
                        if 'right->ad' not in self.edgedic:
                            self.edgedic['right->ad'] = len(self.edgedic)
                        gcode.addEdge(x.id, x.father.child[idx + 1].id, self.edgedic['right->ad'])
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        if 'father->child' not in self.edgedic:
                            self.edgedic['father->child'] = len(self.edgedic)
                        gcode.addEdge(x.id, s.id, self.edgedic['father->child'])
                        #nladrow.append(x.id)
                        #nladcol.append(s.id)
                        #nladdata.append(1)
            nl = nltmp
            vars = dataFile[i]['varwithtype']
            tmpvar = []
            tmptype = []
            tmpuseable = []
            for k, x in enumerate(vars):
                tmpvar.append(x[0])
                tmptype.append(x[1])
                tmpuseable.append(x[2] + 5) 
                for p, nodet in enumerate(nl):
                    if nodet == x[0]:
                        if 'comef' not in self.edgedic:
                            self.edgedic['comef'] = len(self.edgedic)
                        if 'usev' not in self.edgedic:
                            self.edgedic['usev'] = len(self.edgedic)
                        gcode.addEdge(p, self.Nl_Len + k, self.edgedic['comef'])
                        gcode.addEdge(self.Nl_Len + k, p, self.edgedic['usev'])
            if 'dtype' not in self.edgedic:
                self.edgedic['dtype'] = len(self.edgedic)
            if 'stype' not in self.edgedic:
                self.edgedic['stype'] = len(self.edgedic)
            for k, v in enumerate(tmpvar):
                for j, v in enumerate(tmpvar):
                    if tmptype[k] == tmptype[j]:
                        if tmptype[k] == 'ptype':
                            if vars[k][3] == vars[j][3]:
                                gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['stype'])
                            else:
                                gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['dtype'])
                        else:         
                            gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['stype'])
                    else:
                        gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['dtype'])
            inputVar.append(self.pad_seq(self.Get_Em(tmpvar, self.Nl_Voc), self.Var_Len))
            inputType.append(self.pad_seq(self.Get_Em(tmptype, self.Nl_Voc), self.Var_Len))
            inputUseable.append(self.pad_seq(tmpuseable, self.Var_Len))
            maxl = max(len(vars), maxl)
            #continue
            nls.append((dataFile[i]['input'], dataFile[i]['varwithtype']))
            if i == 6:
                print(dataFile[i]['varwithtype'])
            inputpos = dataFile[i]['problist']
            #for j in range(len(inputpos)):
            #    inputpos[j] = inputpos[j]
            inputPos.append(self.pad_seq(inputpos, self.Nl_Len))
            inputparent = dataFile[i]['fatherlist']#lines[5 * i + 2].strip().split()
            inputres = dataFile[i]['rule']#lines[5 * i + 1].strip().split()
            #depth = lines[5 * i + 3].strip().split()
            parentname = dataFile[i]['fathername']#lines[5 * i + 4].strip().lower().split()
            for j in range(len(parentname)):
                parentname[j] = parentname[j].lower()
            inputadrow = []
            inputadcol = []
            inputaddata = []
            #inputad = np.zeros([self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])
            inputrule = [self.ruledict["start -> root"]]
            for j in range(len(inputres)):
                inputres[j] = int(inputres[j])
                inputparent[j] = int(inputparent[j]) + 1
                child.setdefault(inputparent[j], []).append(j + 1)
                if inputres[j] >= 3000000:
                    assert(0)
                    #assert(tmpuseable[inputres[j] - 3000000] == 6)
                    inputres[j] = len(self.ruledict) + self.Nl_Len + self.Nl_Len + inputres[j] - 3000000
                    if (inputres[j] - (len(self.ruledict) + self.Nl_Len + self.Nl_Len) >self.Var_Len):
                        print(inputres[j] - (len(self.ruledict) + self.Nl_Len + self.Nl_Len))
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + self.Var_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict) - self.Nl_Len - self.Nl_Len)
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> copyword3'])
                elif inputres[j] >= 2000000:
                    #assert(0)
                    inputres[j] = len(self.ruledict) + inputres[j] - 2000000
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + self.Var_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict))
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> copyword'])
                elif inputres[j] >= 1000000:
                    inputres[j] = len(self.ruledict) + inputres[j] - 1000000 + self.Nl_Len
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + self.Var_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict) - self.Nl_Len)
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> copyword2'])
                else:
                    inputrule.append(inputres[j])
                #assert(inputrule[-1] < 1022)
                #if inputres[j] - len(self.ruledict) >= self.Nl_Len:
                #    print(inputres[j] - len(self.ruledict))
                if j + 1 < self.Code_Len:
                    inputadrow.append(self.Nl_Len + self.Var_Len + j + 1)
                    inputadcol.append(self.Nl_Len + self.Var_Len + inputparent[j])
                    inputaddata.append(1)
                    #inputad[self.Nl_Len + j + 1, self.Nl_Len + inputparent[j]] = 1
            #inputrule = [self.ruledict["start -> Module"]] + inputres
            #depth = self.pad_seq([1] + depth, self.Code_Len)
            inputnls = self.Get_Em(nl, self.Nl_Voc)
            inputNl.append(self.pad_seq(inputnls, self.Nl_Len))
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            inputruleparent = self.pad_seq(self.Get_Em(["start"] + parentname, self.Code_Voc), self.Code_Len)
            inputrulechild = []
            for x in inputrule:
                if x >= len(self.rrdict):
                    inputrulechild.append(self.pad_seq(self.Get_Em(["copyword"], self.Code_Voc), self.Char_Len))
                else:
                    rule = self.rrdict[x].strip().lower().split()
                    inputrulechild.append(self.pad_seq(self.Get_Em(rule[2:], self.Code_Voc), self.Char_Len))

            inputparentpath = []
            for j in range(len(inputres)):
                if inputres[j] in self.rrdict:
                    tmppath = [self.rrdict[inputres[j]].strip().lower().split()[0]]
                    if tmppath[0] != parentname[j].lower() and tmppath[0] == 'statements' and parentname[j].lower() == 'root':
                        tmppath[0] = 'root'#print(tmppath, parentname[j].lower())
                    if tmppath[0] != parentname[j].lower() and tmppath[0] == 'start':
                        tmppath[0] = parentname[j].lower()
                    #print(tmppath, parentname[j].lower(), inputres)
                    assert(tmppath[0] == parentname[j].lower())
                else:
                    tmppath = [parentname[j].lower()]
                '''siblings = child[inputparent[j]]
                for x in siblings:
                    if x == j + 1:
                        break
                    tmppath.append(parentname[x - 1])'''
                #print(inputparent[j])
                curr = inputparent[j]
                while curr != 0:
                    if inputres[curr - 1] >= len(self.rrdict):
                        #print(parentname[curr - 1].lower())
                        rule = 'root'
                        #assert(0)
                    else:
                        rule = self.rrdict[inputres[curr - 1]].strip().lower().split()[0]
                    #print(rule)
                    tmppath.append(rule)
                    curr = inputparent[curr - 1]
                #print(tmppath)
                inputparentpath.append(self.pad_seq(self.Get_Em(tmppath, self.Code_Voc), 10))
            #assert(0)
            inputrule = self.pad_seq(inputrule, self.Code_Len)
            inputres = self.pad_seq(inputres, self.Code_Len)
            tmp = [self.pad_seq(self.Get_Em(['start'], self.Code_Voc), 10)] + inputparentpath
            inputrulechild = self.pad_list(tmp, self.Code_Len, 10)
            inputRuleParent.append(inputruleparent)
            inputRuleChild.append(inputrulechild)
            inputRes.append(inputres)
            inputRule.append(inputrule)
            inputparent = [0] + inputparent
            inputad = sparse.coo_matrix((inputaddata, (inputadrow, inputadcol)), shape=(self.Nl_Len + self.Var_Len + self.Code_Len, self.Nl_Len + self.Var_Len + self.Code_Len))
            inputParent.append(inputad)
            inputParentPath.append(self.pad_list(inputparentpath, self.Code_Len, 10))
            nlad = sparse.coo_matrix((gcode.val, (gcode.row, gcode.col)), shape=(self.Nl_Len + self.Var_Len, self.Nl_Len + self.Var_Len))
            inputNlad.append(nlad)
        print(maxl)
        batchs = [inputNl, inputNlad, inputRule, inputRuleParent, inputRuleChild, inputRes, inputParent, inputParentPath, inputPos, inputVar, inputType, inputUseable]
        self.data = batchs
        self.nl = nls
        #self.code = codes
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("nl.pkl", "wb").write(pickle.dumps(nls))
            open('edge.pkl', 'wb').write(pickle.dumps(self.edgedic))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("valnl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "test":
            open("testdata.pkl", "wb").write(pickle.dumps(batchs))
            #open("testcode.pkl", "wb").write(pickle.dumps(self.code))
            open("testnl.pkl", "wb").write(pickle.dumps(self.nl))
            #open("testvar.pkl", 'wb').write(pickle.dumps(vardics))
        return batchs

    def __getitem__(self, offset):
        ans = []
        '''if self.dataName == "train":
            h5f = h5py.File("data.h5", 'r')
        if self.dataName == "val":
            h5f = h5py.File("valdata.h5", 'r')
        if self.dataName == "test":
            h5f = h5py.File("testdata.h5", 'r')'''
        for i in range(len(self.data)):
            d = self.data[i][offset]
            if i == 1 or  i == 6:
                tmp = d.toarray().astype(np.int32)
                ans.append(tmp)
            else:
                ans.append(np.array(d))
            '''if i == 6:
                #print(self.data[i][offset])
                tmp = np.eye(self.Code_Len)[d]
                #print(tmp.shape)
                tmp = np.concatenate([tmp, np.zeros([self.Code_Len, self.Code_Len])], axis=0)[:self.Code_Len,:]#self.pad_list(tmp, self.Code_Len, self.Code_Len)
                ans.append(np.array(tmp))
            else:'''
        return ans
    def __len__(self):
        return len(self.data[0])
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.sibiling = None
    def getchildbyname(self, name):
        for x in self.child:
            if x.name == name:
                return x
        return None
    
#dset = SumDataset(args)
