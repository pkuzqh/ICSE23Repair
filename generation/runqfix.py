import sys
sys.path.append('.')
import torch
from torch import optim
from Dataset import SumDataset, Graph
import os
from tqdm import tqdm
from model.Model import *
import numpy as np
import wandb
from copy import deepcopy
import pickle
from ScheduledOptim import *
import sys
from Searchnode import Node
import json
import torch.nn.functional as F
import traceback
from apex import amp
import copy
#from pythonBottom.run import finetune
#from pythonBottom.run import pre
#wandb.init("sql")
from memory_profiler import profile
from line_profiler import LineProfiler
import random
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':500,
    'CodeLen':60,
    'batch_size':96,
    'tbsize':4,
    'embedding_size':256,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'num_steps':50,
    'rulenum':10,
    'cnum':695,
    'use_apex':False,
    'mask_value':-1e9,
    'gradient_accumulation_steps':1,
    'max_grad_norm':5,
    'seed':19970316,
    'varlen':45,
    'mask_id':-1
})
os.environ["CUDA_VISIBLE_DEVICES"]="4,3,6,5"#"0, 2, 4, 5"
#os.environ['CUDA_LAUNCH_BLOCKING']="4"
def save_model(model, dirs='checkpointSearch/', optimizer=None, amp=None):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    '''if args.use_apex:
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'amp':amp.state_dict()
        }
        torch.save(checkpoint, dirs + 'best_model.ckpt')
    else:'''
    torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt'))
def load_model_apex(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt')['model'])
use_cuda = True#torch.cuda.is_available()
onelist = ['root', 'body', 'statements', 'block', 'arguments', 'initializers', 'parameters', 'case', 'cases', 'selectors']
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans
def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    return ans
def getRulePkl(vds):
    inputruleparent = []
    inputrulechild = []
    for i in range(args.cnum):
        rule = vds.rrdict[i].strip().lower().split()
        inputrulechild.append(vds.pad_seq(vds.Get_Em(rule[2:], vds.Code_Voc), vds.Char_Len))
        inputruleparent.append(vds.Code_Voc[rule[0].lower()])
    return np.array(inputruleparent), np.array(inputrulechild)
def getAstPkl(vds):
    rrdict = {}
    for x in vds.Code_Voc:
        rrdict[vds.Code_Voc[x]] = x
    inputchar = []
    for i in range(len(vds.Code_Voc)):
        rule = rrdict[i].strip().lower()
        inputchar.append(vds.pad_seq(vds.Get_Char_Em([rule])[0], vds.Char_Len))
    return np.array(inputchar)
def evalacc(model, dev_set):
    antimask = gVar(getAntiMask(args.CodeLen))
    a, b = getRulePkl(dev_set)
    tmpast = getAstPkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=len(dev_set),
                                              shuffle=False, drop_last=True, num_workers=1)
    model = model.eval()
    accs = []
    tcard = []
    loss = []
    ploss = []
    antimask2 = antimask.unsqueeze(0).repeat(len(dev_set), 1, 1).unsqueeze(1)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.tbsize, 1, 1)
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(args.tbsize, 1).long()
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        with torch.no_grad():
            l, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7], devBatch[8], devBatch[9], devBatch[10], devBatch[11], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, devBatch[5])
            ploss.append(l)
            print(l)
            loss.append(l.max().item())
            pred = pre.argmax(dim=-1)
            resmask = torch.gt(devBatch[5], 0)
            acc = (torch.eq(pred, devBatch[5]) * resmask).float()#.mean(dim=-1)
            #predres = (1 - acc) * pred.float() * resmask.float()
            accsum = torch.sum(acc, dim=-1)
            resTruelen = torch.sum(resmask, dim=-1).float()
            cnum = torch.eq(accsum, resTruelen).sum().float()
            #print((torch.eq(accsum, resTruelen)))
            acc = acc.sum(dim=-1) / resTruelen
            accs.append(acc.mean().item())
            tcard.append(cnum.item())
                        #print(devBatch[5])
                        #print(predres)
    exit(0)
    tnum = np.sum(tcard)
    acc = np.mean(accs)
    l = np.max(loss)

    return acc, tnum, l
def pretrain():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    opt_level = 'O1'
    train_set = PreSumDataset(args, "train")
    print(len(train_set.rrdict))
    print(train_set.ruledict['start -> copyword3'])
    args.mask_id = train_set.Nl_Voc['[mask]']
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.tbsize, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = getAstPkl(train_set)
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(train_set.Code_Voc))).unsqueeze(0).repeat(args.tbsize, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    #dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    print(len(test_set))
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
    model = Decoder(args)
    #load_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0
    maxL = 1e10
    if use_cuda:
        if args.use_apex:
            model = model.cuda()
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        else:
            model = model.cuda()
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        #model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    antimask = gVar(getAntiMask(args.CodeLen))
    #model.to()
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):

            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss = model(dBatch[0], dBatch[1], dBatch[3], dBatch[4], dBatch[5], dBatch[6], tmpindex2, tmpchar, dBatch[2])
            loss = torch.mean(loss)# + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            #optimizer.zero_grad()
            if j % 100 == 0:
                closs = loss.item()
                
                if closs < maxL:
                    print('find better loss %f'%closs)
                    maxL = closs
                    save_model(model)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
                
            if args.use_apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            j += 1
def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    opt_level = 'O1'
    train_set = SumDataset(args, "train")
    print(len(train_set.rrdict))
    print(train_set.ruledict['start -> copyword3'])

    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.tbsize, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = getAstPkl(train_set)
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(train_set.Code_Voc))).unsqueeze(0).repeat(args.tbsize, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    #dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    print(len(test_set))
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
    model = Decoder(args)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0
    maxL = 1e10
    load_model(model, 'checkModel/')
    if use_cuda:
        if args.use_apex:
            model = model.cuda()
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        else:
            model = model.cuda()
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        #model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    #load_model(model)
    antimask = gVar(getAntiMask(args.CodeLen))
    #model.to()
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):
            #print(j)
            if j % 3000 == 0 or j % 3000 == 900 or j % 3000 == 300 or j % 3000 == 600:
                #acc, tnum = evalacc(model, dev_set)
                acc2, tnum2, l = evalacc(model, test_set)
                #print("for dev " + str(acc) + " " + str(tnum) + " max is " + str(maxC))
                print("for test " + str(acc2) + " " + str(tnum2) + " max is " + str(maxC2) + "loss is " + str(l))
                #exit(0)
                if maxL > l:#if maxC2 < tnum2 or maxC2 == tnum2 and maxAcc2 < acc2:
                    maxC2 = tnum2
                    #maxAcc2 = acc2
                    maxL = l
                    print("find better acc " + str(maxAcc2))
                    save_model(model.module, 'checkModel/')
                if maxAcc2 < acc2:
                    maxAcc2 = acc2
                    print("find better acc " + str(maxAcc2))
                    save_model(model.module, 'checkpointAcc/')
            antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[6], dBatch[7], dBatch[8], dBatch[9], dBatch[10], dBatch[11], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, dBatch[5])
            loss = torch.mean(loss)# + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            #optimizer.zero_grad()
            #print(loss)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
                
            if args.use_apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            j += 1
'''class Node:
    def __init__(self, name, d):
        self.name = name
        self.id = d
        self.father = None
        self.child = []
        self.sibiling = None
        self.expanded = False
        self.fatherlistID = 0
        self.fname = ""
    def printTree(self, r):
        s = r.name + " "#print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s'''
def turnold2new2(root):
    if True:
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
                      if len(s.child) == 1:#normal type
                        t = s.child[0].child[0].child[0].name
                        break
                      elif len(s.child) == 2:#array type
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
                      if len(s.child) == 1:
                        t = s.child[0].child[0].child[0].name
                        break
                      elif len(s.child) == 2:
                        t = s.child[0].child[0].child[0].name + '[]'
                        break
                #print(root.printTree(x[1]))
                if t == -1:
                  continue
                typedic[x[0]] = t
    for x in typedic:
        #print(typedic[x])
        if typedic[x] in ['int_ter', 'double_ter', 'long_ter']:
            typedic[x] = 'numeric'
        elif typedic[x] in ['boolean_ter']:
            typedic[x] = 'boolean'
        elif 'String' in typedic[x]:
            typedic[x] = 'string'
        else:
            typedic[x] = 'ptype'
    return vardic, typedic
def getLocVar(node):
  varnames = []
  if node.name == 'VariableDeclarator':
    currnode = -1
    for x in node.child:
      if x.name == 'name' or x.name == 'loc1':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'FormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name' or x.name == 'loc1':
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
def setvar(root, varn):
    if root.name == varn and root.father.name in ['member', 'qualifier']:
        croot = root
        while croot:
            croot.copyable = False
            croot = croot.father
    for x in root.child:
        setvar(x, varn)
def cmpp(ori, target):
    if len(ori) != len(target):
        return False
    for i in range(len(ori)):
        if ori[i] == 'unknown':
            continue
        if ori[i] != target[i]:
            return False
    return True
class SearchNode:
    def deepcopy(self):
        ans = copy.copy(self)
        ans.root = pickle.loads(pickle.dumps(self.root))#copy.deepcopy(self.root)
        ans.state = pickle.loads(pickle.dumps(self.state))#copy.deepcopy(self.state)
        ans.inputparent = pickle.loads(pickle.dumps(self.inputparent))# copy.deepcopy(self.inputparent)
        ans.parent = pickle.loads(pickle.dumps(self.parent))#copy.deepcopy(self.parent)
        ans.everTreepath = pickle.loads(pickle.dumps(self.everTreepath))#copy.deepcopy(self.everTreepath)
        ans.expanded = self.selcetNode(ans.root)#copy.deepcopy(self.expanded)
        return ans
    def __init__(self, ds, nl, classes, method):
        self.methodname = method[:-4]
        self.state = [ds.ruledict["start -> root"]]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.root = Node("root", 2)
        self.inputparent = ["root"]
        self.finish = False
        self.unum = 0
        self.parent = Graph()#np.zeros([args.NlLen + args.CodeLen + args.varlen, args.NlLen + args.CodeLen + args.varlen])
        self.parent.rowNum = args.NlLen + args.CodeLen + args.varlen
        self.parent.colNum = args.NlLen + args.CodeLen + args.varlen
        #self.parent[args.NlLen]
        self.expanded = None
        #self.ruledict = ds.rrdict
        self.expandedname = []
        self.depth = [1]
        for x in ds.ruledict:
            self.expandedname.append(x.strip().split()[0])
        self.var = nl[1]
        self.classes = classes
        nl = nl[0]
        root = Node(nl[0], 0)
        idx = 1
        self.idmap = {}
        self.idmap[0] = root
        currnode = root
        self.actlist = []
        for x in nl[1:]:
            if x != "^":
                nnode = Node(x, idx)
                self.idmap[idx] = nnode
                idx += 1
                nnode.father = currnode
                currnode.child.append(nnode)
                currnode = nnode
            else:
                currnode = currnode.father
        for x in self.var:
            if x[2] == 0:
                setvar(root, x[0])
        #print(self.printTree(root))
        #print(self.var)
        #self.var, self.typedic = turnold2new(root)
        self.everTreepath = []
    def selcetNode(self, root):
        if not root.expanded and root.name in self.expandedname and root.namewithouttype not in onelist: #and self.state[root.fatherlistID] < len(self.ruledict):
            return root
        else:
            for x in root.child:
                ans = self.selcetNode(x)
                if ans:
                    return ans
            if root.namewithouttype in onelist and root.expanded == False:
                return root
        return None
    def selectExpandedNode(self):
        self.expanded = self.selcetNode(self.root)
    def getRuleEmbedding(self, ds, nl):
        inputruleparent = []
        inputrulechild = []
        for x in self.state:
            if x >= len(ds.rrdict):
                inputruleparent.append(ds.Get_Em(["value"], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(["copyword"], ds.Code_Voc), ds.Char_Len))
            else:
                rule = ds.rrdict[x].strip().lower().split()
                #print(rule[0])
                inputruleparent.append(ds.Get_Em([rule[0]], ds.Code_Voc)[0])
                #print(ds.Get_Em([rule[0]], ds.Code_Voc))
                inputrulechild.append(ds.pad_seq(ds.Get_Em(rule[2:], ds.Code_Voc), ds.Char_Len))
        tmp = [ds.pad_seq(ds.Get_Em(['start'], ds.Code_Voc), 10)] + self.everTreepath
        inputrulechild = ds.pad_list(tmp, ds.Code_Len, 10)
        inputrule = ds.pad_seq(self.state, ds.Code_Len)
        #inputrulechild = ds.pad_list(inputrulechild, ds.Code_Len, ds.Char_Len)
        inputruleparent = ds.pad_seq(inputruleparent, ds.Code_Len)
        inputdepth = ds.pad_list(self.depth, ds.Code_Len, 40)
        #print(inputruleparent)
        return inputrule, inputrulechild, inputruleparent, inputdepth
    def getTreePath(self, ds):
        tmppath = [self.expanded.name.lower()]
        node = self.expanded.father
        while node:
            tmppath.append(node.name.lower())
            node = node.father
        tmp = ds.pad_seq(ds.Get_Em(tmppath, ds.Code_Voc), 10)
        self.everTreepath.append(tmp)
        return ds.pad_list(self.everTreepath, ds.Code_Len, 10)
    def checkapply(self, rule, ds):
        if rule >= len(ds.ruledict):
            if self.expanded.name == 'root' and len(self.expanded.child) == 0:
                return False
            if self.expanded.name == 'root' and rule - len(ds.ruledict) >= args.NlLen:
                if rule - len(ds.ruledict) - args.NlLen not in self.idmap:
                    return False
                if self.idmap[rule - len(ds.ruledict) - args.NlLen].namewithouttype not in ['MemberReference', 'BasicType', 'operator', 'qualifier', 'member', 'Literal']:
                    return False
                if '.0' in self.idmap[rule - len(ds.ruledict) - args.NlLen].getTreestr():
                    return False
                #print(self.idmap[rule - len(ds.ruledict)].name)
                #assert(0)
                return True
            if rule - len(ds.ruledict) >= 2 * args.NlLen:
                idx = rule - len(ds.ruledict) - 2 * args.NlLen
                if self.expanded.namewithouttype not in ['member', 'value', 'name', 'qualifier']:
                    return False
                if idx >= len(self.var):
                    print(self.var, idx)
                    return False
                if self.var[idx][2] == 0:
                    return False
                if self.expanded.namewithouttype in ['qualifier']:
                    if self.var[idx][1] != 'ptype':
                        return False
                if self.expanded.namewithouttype in ['member', 'value']:
                    ttype = ''
                    if len(self.expanded.name.split('🚀')) > 1:
                        ttype = self.expanded.name.split('🚀')[1]
                    if ttype != '':
                        if ttype != self.var[idx][1]:
                            return False
                return True
            if rule - len(ds.ruledict) >= args.NlLen:
                return False
            idx = rule - len(ds.ruledict)
            if idx not in self.idmap:
                return False
            if self.idmap[idx].name != self.expanded.name:
                #print(self.idmap[idx].name, self.expanded.name, idx)
                return False
            if not self.idmap[idx].copyable:
                return False
            #if 'excerpt_ter' in self.idmap[idx].getTreestr():
            #    return False
            if self.idmap[idx].name == 'member' and self.expanded.father.name == 'MethodInvocation':
                hasqualifier = False
                param = []
                for child in self.expanded.father.child:
                    if child.name == 'qualifier':
                        hasqualifier = True
                        break
                    if child.name == 'arguments':
                        for y in child.child:
                            mtype = y.name.split('🚀')
                            if len(mtype) > 1:
                                param.append(mtype[1])
                            else:
                                param.append('unkown')
                if not hasqualifier:
                    methodname = self.idmap[idx].child[0].name[:-4]
                    try:
                        for m in self.classes['classes'][0]['methods']:
                            cmpparams = []
                            if m['name'] == methodname:
                                if 'arguments' in m:
                                    for arggs in m['arguments']:
                                        if arggs['type'] in ['int', 'float', 'double', 'long']:
                                            cmpparams.append('numeric')
                                        elif arggs['type'] in ['String', 'StringBuffer', 'StringBuilder']:
                                            cmpparams.append('string')
                                        elif arggs['type'] in ['boolean']:
                                            cmpparams.append('bool')
                                        else:
                                            cmpparams.append('unkown')
                            if cmpp(param, cmpparams):
                                return True
                    except:
                        return False
                    return False
                return True
        else:
            rules = ds.rrdict[rule]
            if rules == 'start -> unknown':
                if self.expanded.name == 'root':
                    return False
                if self.unum >= 1:
                    return False
                if self.expanded.name == 'member' and self.expanded.father.name == 'MethodInvocation':
                    hasqualifier = False
                    param = []
                    for child in self.expanded.father.child:
                        if child.name == 'qualifier':
                            hasqualifier = True
                            break
                        if child.name == 'arguments':
                            for y in child.child:
                                mtype = y.name.split('🚀')
                                if len(mtype) > 1:
                                    param.append(mtype[1])
                                else:
                                    param.append('unkown')
                    if not hasqualifier:
                        try:
                            for m in self.classes['classes'][0]['methods']:
                                cmpparams = []
                                arglist = [] if 'params' not in m else m['params']
                                if True:
                                    for arggs in arglist:
                                        if arggs['type'] in ['int', 'float', 'double', 'long']:
                                            cmpparams.append('numeric')
                                        elif arggs['type'] in ['String', 'StringBuffer', 'StringBuilder']:
                                            cmpparams.append('string')
                                        elif arggs['type'] in ['boolean']:
                                            cmpparams.append('bool')
                                        else:
                                            cmpparams.append('unkown')
                                if cmpp(param, cmpparams):
                                    return True
                        except:
                            return False
                        return False
                return True
            if self.expanded.name == 'root' and len(self.expanded.child) == 0:
                if rules not in ['root -> add', 'root -> modified']:
                    return False
            lst = rules.split()
            if ('member' in lst[0])  or 'qualifier' in lst[0]:
                if 'par' in lst[-1] or 'loc' in lst[-1]:
                    var = lst[-1]
                    ttst =lst[0].split('🚀')
                    if len(ttst) > 1:
                        mtype = ttst[1]
                    else:
                        mtype = 'ptype'
                    invar = False
                    for x in self.var:
                        if var == x[0]:
                            if mtype == 'ptype':
                                invar = True
                                break
                            if x[1] == mtype:
                                invar = True
                                break
                    if not invar:
                        return False 
            if 'member -> par' in rules or 'member -> loc' in rules or 'qualifier -> par' in rules or 'qualifier -> loc' in rules:
                var = rules.strip().split()[2]
                invar = False
                for x in self.var:
                    if var == x[0]:
                        invar = True
                        break
                if not invar:
                    return False 
            #if len(self.depth) == 1:
                #print(rules)
            #    if rules != 'root -> modified' or rules != 'root -> add':
            #        return False
            if rules.strip().split()[0].lower() != self.expanded.name.lower():
                return False
            if self.expanded.namewithouttype == 'member' and rules == 'member -> meth0':
                #have no qualifier
                return True
                if self.expanded.father.name == 'MethodInvocation' and self.expanded.father.child[0].name == 'arguments':
                    param = []
                    for x in self.expanded.father.child:
                        if x.namewithouttype == 'arguments':
                            for y in x.child:
                                mtype = y.name.split('🚀')
                                if len(mtype) == 1:
                                    param.append('unknown')
                                else:
                                    param.append(mtype[1])
                            break
                    try:
                        for m in self.classes['classes'][0]['methods']:
                            cmpparam = []
                            if m['name'] == self.methodname:
                                for arggs in m['params']:
                                    if arggs['type'] in ['float', 'double', 'int']:
                                        cmpparam.append('numeric')
                                    elif arggs['type'] in ['String']:
                                        cmpparam.append('string')
                                    elif arggs['type'] in ['boolean']:
                                        cmpparam.append('bool')
                                    else:
                                        cmpparam.append('unknown')
                                if param == cmpparam:
                                    return True
                    except:
                        return False
                    return False
                    
                    
            '''if self.expanded.namewithouttype in ['member', 'value']:
                lst = rules.strip().split()
                if len(lst) == 3 and ('par' in lst[2] or 'loc' in lst[2]):
                    #print(self.var, self.expanded.name)
                    if lst[2] not in self.var:
                        return False
                    ttype = ''
                    if len(self.expanded.name.split('🚀')) > 1:
                        ttype = self.expanded.name.split('🚀')[1]
                    if ttype != '':
                        if ttype != self.typedic[lst[2]]:
                            return False
                    #print(self.printTree(self.root), rules, 1)'''
        return True
    def copynode(self, newnode, original):
        for x in original.child:
            nnode = Node(x.name, -1)
            nnode.father = newnode
            nnode.expanded = True
            newnode.child.append(nnode)
            self.copynode(nnode, x)
        return
    def applyrule(self, rule, ds):
        '''if rule < len(ds.ruledict):
            print(rule, ds.rrdict[rule])
        elif rule >= len(ds.ruledict) + args.NlLen:
            print('copy', self.idmap[rule - len(ds.ruledict) - args.NlLen].name)
        else:
            print('copy2', self.idmap[rule - len(ds.ruledict)].name)'''
        if rule >= len(ds.ruledict):
            if rule >= len(ds.ruledict) + 2 * args.NlLen:
                idx = rule - (len(ds.ruledict) + 2 * args.NlLen)
                self.actlist.append('copy-' + self.var[idx][0])
            else:
                if rule >= len(ds.ruledict) + args.NlLen:
                    idx = rule - len(ds.ruledict) - args.NlLen
                else:
                    idx = rule - len(ds.ruledict)
                self.actlist.append('copy-' + self.idmap[idx].name)
        else:
            self.actlist.append(ds.rrdict[rule])
        if rule >= len(ds.ruledict):
            nodesid = rule - len(ds.ruledict)
            if nodesid >= 2 * args.NlLen:
                nodesid = nodesid - 2 * args.NlLen
                nnode = Node(self.var[nodesid][0], nodesid)
                nnode.fatherlistID = len(self.state)
                nnode.father = self.expanded
                nnode.fname = "-" + (self.var[nodesid][0])
                self.expanded.child.append(nnode)
            elif nodesid >= args.NlLen:
                nodesid = nodesid - args.NlLen
                nnode = Node(self.idmap[nodesid].name, nodesid)
                nnode.fatherlistID = len(self.state)
                nnode.father = self.expanded
                nnode.fname = "-" + self.printTree(self.idmap[nodesid]) + str(nodesid)
                self.expanded.child.append(nnode)
            else:
                nnode = self.idmap[nodesid]
                if nnode.name == self.expanded.name:
                    self.copynode(self.expanded, nnode)
                    nnode.fatherlistID = len(self.state)
                else:
                    if nnode.name == 'VariableDeclarator':
                        currnode = -1
                        for x in nnode.child:
                            if x.name == 'name':
                                currnode = x
                                break
                        nnnode = Node(currnode.child[0].name, -1)
                    else:
                        currnode = -1
                        for x in nnode.child:
                            if x.name == 'name':
                                currnode = x
                                break
                        nnnode = Node(currnode.child[0].name, -1)
                    nnnode.father = self.expanded
                    self.expanded.child.append(nnnode)
                    nnnode.fatherlistID = len(self.state)
                self.expanded.expanded = True
        else:
            rules = ds.rrdict[rule]
            if rules == 'start -> unknown':
                self.unum += 1
            #if rules.strip().split()[0] != self.expanded.name:
            #    #print(self.expanded.name)
            #    assert(0)
            #    return False
            #assert(rules.strip().split()[0] == self.expanded.name)
            if rules.strip() == self.expanded.name + " -> End":
                self.expanded.expanded = True
            else:
                for x in rules.strip().split()[2:]:
                    nnode = Node(x, -1)                   
                    #nnode = Node(x, self.expanded.depth + 1)
                    self.expanded.child.append(nnode)
                    nnode.father = self.expanded
                    nnode.fatherlistID = len(self.state)
        #self.parent.append(self.expanded.fatherlistID)
        #print(self.expanded.fatherlistID)
        self.parent.addEdge(args.NlLen + args.varlen + len(self.inputparent), args.NlLen + args.varlen + self.expanded.fatherlistID, 1)
        #[args.NlLen + args.varlen + len(self.inputparent), args.NlLen + args.varlen + self.expanded.fatherlistID] = 1
        if rule >= len(ds.ruledict) + 2 * args.NlLen:
            idx = rule - (len(ds.ruledict) + 2 * args.NlLen)
            self.parent.addEdge(args.NlLen + args.varlen + len(self.inputparent), args.NlLen + idx, 1)
            #self.parent[args.NlLen + args.varlen + len(self.inputparent), args.NlLen + idx] = 1
        elif rule >= len(ds.ruledict) + args.NlLen:
            self.parent.addEdge(args.NlLen + args.varlen + len(self.inputparent), rule - len(ds.ruledict) - args.NlLen, 1)
            #self.parent[args.NlLen + args.varlen + len(self.inputparent), rule - len(ds.ruledict) - args.NlLen] = 1
        elif rule >= len(ds.ruledict):
            self.parent.addEdge(args.NlLen + args.varlen + len(self.inputparent), rule - len(ds.ruledict), 1)
            #self.parent[args.NlLen + args.varlen + len(self.inputparent), rule - len(ds.ruledict)] = 1

        if rule >= len(ds.ruledict) + 2 * args.NlLen:
            self.state.append(ds.ruledict['start -> copyword3'])
        elif rule >= len(ds.ruledict) + args.NlLen:
            self.state.append(ds.ruledict['start -> copyword2'])
        elif rule >= len(ds.ruledict):
            self.state.append(ds.ruledict['start -> copyword'])
        else:
            self.state.append(rule)
        #self.state.append(rule)
        self.inputparent.append(self.expanded.name.lower())
        #self.depth.append(1)
        if self.expanded.namewithouttype not in onelist:
            self.expanded.expanded = True
        return True
    def printTree(self, r):
        s = r.name + r.fname + " "#print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s
    def getTreestr(self):
        return self.printTree(self.root)

        
beamss = []
import line_profiler

def BeamSearch(inputnl, vds, model, beamsize, batch_size, k):
    batch_size = len(inputnl[0].view(-1, args.NlLen))
    rrdic = {}
    for x in vds.Code_Voc:
        rrdic[vds.Code_Voc[x]] = x
    print(vds.rrdict[860])
    print(vds.rrdict[851])
    #print(rrdic[183])
    tmpast = getAstPkl(vds)
    a, b = getRulePkl(vds)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    rulead = gVar(pickle.load(open("train/data/rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    tmpindex = gVar(np.arange(len(vds.ruledict))).unsqueeze(0).repeat(2, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(vds.Code_Voc))).unsqueeze(0).repeat(2, 1).long()
    with torch.no_grad():
        beams = {}
        hisTree = {}
        nlencode = {}
        #print(len(vds.nl))
        for i in range(batch_size):
            beams[i] = [SearchNode(vds, vds.nl[args.batch_size * k + i], vds.classes[args.batch_size * k + i], vds.method[args.batch_size * k + i])]
            hisTree[i] = {}
        #nlencode = model.calNlEncode(gVar(inputnl[0]), ginputnl[1], inputnl[8], inputnl[9], inputnl[10], inputnl[11], tmpchar, tmpindex2)
        nlencode = model.calNlEncode(gVar(inputnl[0]), gVar(inputnl[1]), gVar(inputnl[8]), gVar(inputnl[9]), gVar(inputnl[10]), gVar(inputnl[11]), gVar(tmpchar), gVar(tmpindex2))
        index = 0
        antimask = gVar(getAntiMask(args.CodeLen))
        endnum = {}
        continueSet = {}
        tansV = {}
        for lenidx in tqdm(range(args.CodeLen), desc="BeanSearch-batch%d"%k):
            print(index)
            tmpbeam = {}
            ansV = {}
            #for x in beams[0]:
            #    print(x.getTreestr())
            #    print(x.actlist)
            #    print(x.prob)
            #print("kkkkkkkkkkkkk")
            if len(endnum) == batch_size:
                break
            if index >= args.CodeLen:
                break
            for p in range(beamsize):
                tmprule = []
                tmprulechild = []
                tmpruleparent = []
                tmptreepath = []
                tmpAd = []
                validnum = []
                tmpdepth = []
                for i in range(batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    #print('--------------------------', x.printTree(x.root), x.prob, i)
                    x.selectExpandedNode()
                    #if x.expanded != None:
                    #    print(x.expanded.name)
                    #print(x.expanded.name)
                    if x.expanded == None or len(x.state) >= args.CodeLen:
                        x.finish = True
                        #print(x.printTree(x.root), x.prob)
                        ansV.setdefault(i, []).append(x)
                    else:
                        #print(x.expanded.name)
                        validnum.append(i)
                        a, b, c, d = x.getRuleEmbedding(vds, vds.nl[args.batch_size * k + i])
                        tmprule.append(a)
                        tmprulechild.append(b)
                        tmpruleparent.append(c)
                        tmptreepath.append(x.getTreePath(vds))
                        #tmp = np.eye(vds.Code_Len)[x.parent]
                        #tmp = np.concatenate([tmp, np.zeros([vds.Code_Len, vds.Code_Len])], axis=0)[:vds.Code_Len,:]#self.pad_list(tmp, self.Code_Len, self.Code_Len)
                        tmpAd.append(x.parent.tonumpy())
                        tmpdepth.append(d)
                #print("--------------------------")
                if len(tmprule) == 0:
                    continue
                #batch_size = len(tmprule)
                antimasks = antimask.unsqueeze(0).repeat(len(tmprule), 1, 1).unsqueeze(1)
                tmprule = np.array(tmprule)
                tmprulechild = np.array(tmprulechild)
                tmpruleparent = np.array(tmpruleparent)
                tmptreepath = np.array(tmptreepath)
                tmpAd = np.array(tmpAd)
                tmpdepth = np.array(tmpdepth)
                '''print(inputnl[2][0][:index + 1], tmprule[0][:index + 1])
                assert(np.array_equal(inputnl[2][0][:index + 1], tmprule[0][:index + 1]))
                #assert(np.array_equal(inputnl[3][0][:index + 1], tmpruleparent[0][:index + 1]))
                assert(np.array_equal(inputnl[4][0][:index + 1], tmprulechild[0][:index + 1]))
                assert(np.array_equal(inputnl[6][0][:index + 1], tmpAd[0][:index + 1]))
                assert(np.array_equal(inputnl[7][0][:index + 1], tmptreepath[0][:index + 1]))
                #assert(np.array_equal(inputnl[8][0][:index + 1], tmpdepth[0][:index + 1]))'''
                #result = model.forward(nlencode, gVar(inputnl[1][validnum]).view(-1, args.NlLen + args.varlen, args.NlLen + args.varlen), gVar(tmprule), gVar(tmpruleparent), gVar(tmprulechild), gVar(tmpAd), gVar(tmptreepath), gVar(inputnl[8][validnum]).view(-1, args.NlLen), gVar(inputnl[9][validnum]).view(-1, args.varlen), gVar(inputnl[10][validnum]).view(-1, args.varlen), gVar(inputnl[11][validnum]).view(-1, args.varlen), tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimasks, nlencode[0][validnum], nlencode[1][validnum], None, "test")
                result = model.pre_forward(gVar(inputnl[0][validnum]).view(-1, args.NlLen), gVar(inputnl[1][validnum]).view(-1, args.NlLen + args.varlen, args.NlLen + args.varlen), gVar(tmprule), gVar(tmpruleparent), gVar(tmprulechild), gVar(tmpAd), gVar(tmptreepath), gVar(inputnl[8][validnum]).view(-1, args.NlLen), gVar(inputnl[9][validnum]).view(-1, args.varlen), gVar(inputnl[10][validnum]).view(-1, args.varlen), gVar(inputnl[11][validnum]).view(-1, args.varlen), tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimasks, nlencode[0][validnum], nlencode[1][validnum], None, "test")
                results = result#result.data.cpu().numpy()
                #print(result, inputCode)
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    if x.expanded.name == 'MethodInvocation':
                        a = 1
                        pass
                    tmpbeamsize = 0#beamsize
                    result = results[currIndex, index]#np.negative(results[currIndex, index])
                    currIndex += 1
                    cresult = result#np.negative(result)
                    if 'root StatementExpression expression MethodInvocation qualifier otherwise_ter ^ ^ arguments BinaryOperation operator -_ter ^ ^ operandl MemberReference qualifier digit_list_ter ^ ^ member length_ter ^ ^ ^ ^ operandr Literal value 1_ter ^ ^ ^ ^ ^ Literal value' in x.root.printTree(x.root):
                        a = 1
                    if index == 7:
                        a = 1
                    indexs = torch.argsort(result, descending=True)
                    for i in range(len(indexs)):
                        if tmpbeamsize >= 30:
                            break
                        #print(x.prob + torch.log(cresult[indexs[i]]).item())
                        if x.prob + torch.log(cresult[indexs[i]]).item() < -20:
                            break
                        c = x.checkapply(indexs[i].item(), vds)
                        #if c:
                        #    if indexs[i] < len(vds.rrdict):
                        #        print(vds.rrdict[indexs[i]])
                        #    print(x.printTree(x.root), c, 'test', x.prob + np.log(cresult[indexs[i]]))
                        if c:
                            tmpbeamsize += 1
                            #continue
                        else:
                            continue
                        '''copynode = deepcopy(x)
                        #if indexs[i] >= len(vds.rrdict):
                            #print(cresult[indexs[i]])
                        c = copynode.applyrule(indexs[i], vds.nl[args.batch_size * k + j])
                        if not c:
                            tmpbeamsize += 1
                            continue'''
                        #copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        prob = x.prob + np.log(cresult[indexs[i]].item())#copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append([prob, indexs[i].item(), x])
                        #tmpbeam.setdefault(j, []).append(copynode)
                    #print(tmpbeam[0].prob)
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
                    tansV.setdefault(i, []).extend(ansV[i])
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)
                    beams[j] = []
                    for x in tmp:
                        if len(beams[j]) >= beamsize:
                            break
                        if x[1] != -1:
                            copynode = x[2].deepcopy()#pickle.loads(pickle.dumps(x[2]))
                            '''if x[1] >= len(vds.rrdict) + args.CodeLen:
                                print(len(vds.tablename[s[args.batch_size * k + j]['database_id']]['column_names']))
                                if (x[1] - len(vds.rrdict) - args.CodeLen >= 4 and args.batch_size * k + j == 20) or x[1] - len(vds.rrdict) - args.CodeLen >= len(vds.tablename[s[args.batch_size * k + j]['database_id']]['table_names']) + len(vds.tablename[s[args.batch_size * k + j]['database_id']]['column_names']):
                                    print(vds.tabless[args.batch_size * k + j])
                                    print(x[1] - len(vds.rrdict) - args.CodeLen)
                                    assert(0)'''
                            #print(x[1])
                            copynode.applyrule(x[1], vds)
                            if 'par2' in copynode.getTreestr():
                                a = 0
                            #print(x[0])
                            if copynode.getTreestr() in hisTree:
                                continue
                            copynode.prob = x[0]
                            beams[j].append(copynode)
                            hisTree[j][copynode.getTreestr()] = 1
                        else:
                            beams[j].append(x[2])
            #if index >= 2:
            #    assert(0)
            index += 1  
        for i in range(batch_size):
            if i in ansV:
                print(i, ansV[i])
                tansV.setdefault(i, []).extend(ansV[i])
        for j in range(batch_size):
            visit = {}
            tmp = []
            for x in tansV[j]:
                if x.getTreestr() not in visit and x.finish:
                    visit[x.getTreestr()] = 1
                    tmp.append(x)
                else:
                    continue
            beams[j] = sorted(tmp, key=lambda x: x.prob, reverse=True)[:beamsize]  
        #for x in beams:
        #    beams[x] = sorted(beams[x], key=lambda x: x.prob, reverse=True)   
        return beams
        for i in range(len(beams)):
            mans = -1000000
            lst = beams[i]
            tmpans = 0
            for y in lst:
                #print(y.getTreestr())
                if y.prob > mans:
                    mans = y.prob
                    tmpans = y
            beams[i] = tmpans
        #open("beams.pkl", "wb").write(pickle.dumps(beamss))
        return beams
        #return beams
def test():
    #pre()
    #os.environ["CUDA_VISIBLE_DEVICES"]="5, 7"
    dev_set = SumDataset(args, "test", True)
    rulead = gVar(pickle.load(open("train/data/rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = getAstPkl(dev_set)
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(2, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(2, 1).long()
    #print(len(dev_set))
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict) + args.NlLen
    print(dev_set.rrdict[152])
    args.batch_size = 12
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0)
    model = model.eval()
    load_model(model, 'pretrainModel/checkModel/')
    return model
    #return model
    f = open("outval2.txt", "w")
    index = 0 
    indexs = 0
    antimask = gVar(getAntiMask(args.CodeLen))
    antimask2 = antimask.unsqueeze(0).repeat(1, 1, 1).unsqueeze(1)

    for x in tqdm(devloader):
        if indexs < 0:
            indexs += 1
            continue
        #if indexs > 5:
        #    break
        '''pre = model(gVar(x[0]), gVar(x[1]), gVar(x[2]), gVar(x[3]), gVar(x[4]), gVar(x[6]), gVar(x[7]), gVar(x[8]), tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, None, 'test')
        #print(pre[0,3,4020], pre[0,3,317])
        pred = pre.argmax(dim=-1)
        #print(len(dev_set.ruledict))
        print(x[5])
        resmask = torch.gt(gVar(x[5]), 0)
        acc = (torch.eq(pred, gVar(x[5])) * resmask).float()#.mean(dim=-1)
        predres = (1 - acc) * pred.float() * resmask.float()
        accsum = torch.sum(acc, dim=-1)
        resTruelen = torch.sum(resmask, dim=-1).float()
        cnum = (torch.eq(accsum, resTruelen)).sum().float()
        if cnum.item() != 1:
            indexs += 1
            continue'''
        ans = BeamSearch((x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]), dev_set, model, 150, args.batch_size, indexs)
        for i in range(args.batch_size):
            if i not in ans:
                continue
            beam = ans[i]
            f.write(str(indexs * args.batch_size + i))
            for x in beam:
                f.write(x.getTreestr() + " " + str(x.prob) + "\n")
            f.write("-------\n")
            f.flush()
                #print(x.getTreestr())
        indexs += 1
        #exit(0)
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)
def findnodebyid(root, idx):
    if root.id == idx:
        return root
    for x in root.child:
        t = findnodebyid(x, idx)
        if t:
            return t
def getroot(strlst):
    tokens = strlst.split()
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root
def getMember(node):
    for x in node.child:
        if x.name == 'member':
            return x.child[0].name
def setSameid(root, rootsave):
    rootsave.id = root.id
    for i in range(len(root.child)):
        setSameid(root.child[i], rootsave.child[i])
    return
def applyoperater(ans, subroot):
    #print(ans.root.printTree(ans.root))
    #print(subroot.printTree(subroot))
    copynode = getroot(subroot.printTree(subroot))
    setSameid(subroot, copynode)
    #copynode = pickle.loads(pickle.dumps(subroot))
    change = False
    type = ''
    for x in ans.root.child:
        if x.id != -1:
            change = True
            node = findnodebyid(copynode, x.id)
            if node is None:
                continue
            if node.namewithouttype == 'member':
                type = node.child[0].name
                #assert(0)
            elif node.namewithouttype == 'MemberReference':
                type = getMember(node)#node.child[0].child[0].name
                #print(6, type)
            elif node.namewithouttype == 'qualifier':
                type = node.child[0].name
            elif node.namewithouttype == 'operator' or node.namewithouttype == 'Literal' or node.namewithouttype == 'BasicType':
                type = 'valid'
            else:
                #print(node.name)
                assert(0)
            #print(x.printTreeWithoutType(x))
            #assert(0)
            #print(node.name)
            idx = node.father.child.index(node)
            node.father.child[idx] = x
            x.father = node.father
    if change:
        node = Node('root', -1)
        node.child.append(copynode)
        copynode.father = node
        ans.solveroot = node#copynode
        ans.type = type
        #print(node.printTree(ans.solveroot))
    else:
        ans.solveroot = ans.root
        ans.type = type
    #print(copynode.printTree(copynode))
    #assert(0)
    return
import re 
def replaceVar(root, rrdict):
    if root.name in rrdict:
        root.name = rrdict[root.name]
        root.namewithouttype = root.name
    elif len(root.child) == 0:
        if re.match('loc%d', root.name) is not None or re.match('par%d', root.name) is not None:
            return False
    ans = True
    for x in root.child:
        ans = ans and replaceVar(x, rrdict)
    return ans
def getUnknown(root):
    if root.name == 'unknown':
        return [root]
    ans = []
    for x in root.child:
        ans.extend(getUnknown(x))
    return ans
def solveUnknown(ans, vardic, typedic, classcontent, sclassname, mode):
    nodes = getUnknown(ans.solveroot)
    fans =  []
    if len(nodes) >= 2:
        return []
    elif len(nodes) == 0:
        #print(ans.root.printTree(ans.solveroot))
        return [ans.root.printTreeWithoutType(ans.solveroot)]
    else:
        #print(2)
        unknown = nodes[0]
        if unknown.father.father and unknown.father.father.namewithouttype == 'MethodInvocation':
            classname = ''
            args = []
            print('method')
            if unknown.father.namewithouttype == 'member':
                for x in unknown.father.father.child:
                    if x.namewithouttype == 'qualifier':
                        print(x.child[0].name, typedic)
                        if x.child[0].namewithouttype in typedic: 
                            classname = typedic[x.child[0].namewithouttype]
                            break
                        else:
                            if sclassname == 'org.jsoup.nodes.Element':
                                sclassname = 'org.jsoup.nodes.Node'
                            for f in classcontent[sclassname + '.java']['classes'][0]['fields']:
                                print(x.child[0].name, f['name'])
                                if f['name'] == x.child[0].namewithouttype[:-4]:
                                     classname = f['type']
                                     break
                for x in unknown.father.father.child:
                    if x.namewithouttype == 'arguments':
                        for y in x.child:
                            if y.namewithouttype == 'MemberReference':
                                try:
                                    if y.child[0].child[0].namewithouttype in typedic:
                                        args.append(typedic[y.child[0].child[0].namewithouttype])
                                    else:
                                        #print(6, y.child[0].child[0].name)
                                        args.append('int')#return []
                                except:
                                    #print('gg2')
                                    return []
                            elif y.namewithouttype == 'Literal':
                                if y.child[0].child[0].namewithouttype == "<string>_ter":
                                    args.append("String")
                                else:
                                    args.append("int")
                            elif y.namewithouttype == 'MethodInvocation':
                                args.append('ArrayList')
                            else:
                                print('except')
                                return []
            print(7, classname)
            if classname == '':
                classbody = classcontent[sclassname + '.java']['classes']
            elif classname != '':
                if classname + ".java" not in classcontent:
                    #print(5, classname )
                    hasclass = False
                    for nclass in classcontent[sclassname + '.java']['classes']:
                        if nclass['name'] == classname:
                            classbody = [nclass]
                            hasclass = True
                    if not hasclass:
                        return []
                else:
                    classbody = classcontent[classname + '.java']['classes']
            #print(5, sclassname, classbody, classname)
            #print(8)
            if unknown.father.namewithouttype == 'qualifier':
                vtype = ""
                for x in classbody[0]['fields']:
                    #print(x)
                    if x['name'] == ans.type[:-4]:
                        vtype = x['type']
                        break
            if 'IfStatement' in ans.getTreestr():
                if mode == 1 and len(ans.solveroot.child) == 1 and classname != 'Node':
                    #print(ans.solveroot.printTree(ans.solveroot))
                    return []
                #print(ans.solveroot.printTree(ans.solveroot))
                if unknown.father.namewithouttype == 'member':
                    for x in classbody[0]['methods']:
                        if len(x['params']) == 0 and x['type'] == 'boolean':
                            unknown.namewithouttype = x['name'] + "_ter"
                                #print('gggg', unknown.printTree(ans.solveroot))
                            fans.append(unknown.printTreeWithoutType(ans.solveroot))
                elif unknown.father.namewithouttype == 'qualifier':
                    for x in classbody[0]['fields']:
                        if x['type'] == vtype:
                            unknown.namewithouttype = x['name'] + "_ter"
                            fans.append(unknown.printTreeWithoutType(ans.solveroot))
            else:
                #print("a", args)
                if mode == 0 and ans.root == ans.solveroot and len(args) == 0 and classname != 'EndTag' and 'WhileStatement' not in ans.getTreestr():
                    return  [] 
                otype = ""
                if 'WhileStatement' in ans.getTreestr() and unknown.father.father.father and unknown.father.father.father.namewithouttype == 'condition':
                    otype = 'boolean'
                
                if classname == 'EndTag':
                    otype = "String"
                
                if mode == 0 and ans.type != '':
                    args = []

                    if ans.type == "valid":
                        return []
                    for m in classbody[0]['methods']:
                            #print(m['name'])
                        if m['name'] == ans.type[:-4]:
                            otype = m['type']
                            for y in m['params']:
                                args.append(y['type'])
                            break
                #print(args, ans.type, 'o')
                if unknown.father.namewithouttype == 'member':
                    #print(mode, ans.type, args)
                    for x in classbody[0]['methods']:
                        #print(x)
                        #if 'type' in x:
                        #    print(x['type'], otype, x['name'], ans.type)
                        if len(args) == 0 and len(x['params']) == 0:
                            if mode == 0 and x['type'] != otype:
                                continue
                            if mode == 1 and x['type'] is not None and 'Assignment' not in ans.solveroot.getTreestr():
                                continue
                            if ('min' in ans.type.lower() and 'max' in  x['name'].lower()) or ('max' in ans.type.lower() and 'min' in  x['name'].lower()):
                                continue
                            #if mode == 1 and x['type'] != "null":
                            #    continue
                            unknown.namewithouttype = x['name'] + "_ter"
                            #print('gggg', unknown.printTree(ans.solveroot))
                            fans.append(unknown.printTreeWithoutType(ans.solveroot))
                        #print(x['name'], x['type'], args)
                        if ans.type != '':
                            if mode == 0 and len(args) > 0 and x['type'] == otype:
                                targ = []
                                for y in x['params']:
                                    targ.append(y['type'])
                                if args == targ:
                                    if ('min' in ans.type.lower() and 'max' in  x['name'].lower()) or ('max' in ans.type.lower() and 'min' in  x['name'].lower()):
                                        continue
                                    unknown.namewithouttype = x['name'] + "_ter"
                                    fans.append(unknown.printTreeWithoutType(ans.solveroot))
                        else:
                            #print(10)
                            if mode == 0 and len(args) > 0:
                                #print(11)
                                targ = []
                                for y in x['params']:
                                    targ.append(y['type'])
                                #print('p', targ, x['name'], x)
                                if args == targ and 'type' in x and x['type'] is None:
                                    unknown.namewithouttype = x['name'] + "_ter"
                                    fans.append(unknown.printTreeWithoutType(ans.solveroot))
                elif unknown.father.namewithouttype == 'qualifier':
                    if ans.type == 'valid':
                        return []
                    if 'fields' not in classbody[0]:
                        return []
                    for x in classbody[0]['fields']:
                        if x['type'] == vtype:
                            unknown.namewithouttype = x['name'] + "_ter"
                            fans.append(unknown.printTreeWithoutType(ans.solveroot))
                    for x in classbody[0]['methods']:
                        if x['type'] == vtype and len(x['params']) == 0:
                            tmpnode = Node('MethodInvocation', -1)
                            tmpnode1 = Node('member', -1)
                            tmpnode2 = Node(x['name'] + "_ter", -1)
                            tmpnode.child.append(tmpnode1)
                            tmpnode1.father = tmpnode
                            tmpnode1.child.append(tmpnode2)
                            tmpnode2.father = tmpnode1
                            unknown.namewithouttype = " ".join(tmpnode.printTreeWithoutType(tmpnode).split()[:-1])#tmpnode.printTree(tmpnode)
                            fans.append(unknown.printTreeWithoutType(ans.solveroot))
        elif unknown.father.namewithouttype == 'qualifier':
            classbody = classcontent[sclassname + '.java']['classes']
            vtype = ""
            for x in classbody[0]['fields']:
                if x['name'] == ans.type[:-4]:
                    vtype = x['type']
                    break
            #print(5, vtype)
            for x in classbody[0]['fields']:
                if x['type'] == vtype:
                    unknown.namewithouttype = x['name'] + "_ter"
                    fans.append(unknown.printTreeWithoutType(ans.solveroot))
            for x in classbody[0]['methods']:
                if x['type'] == vtype and len(x['params']) == 0:
                    tmpnode = Node('MethodInvocation', -1)
                    tmpnode1 = Node('member', -1)
                    tmpnode2 = Node(x['name'] + "_ter", -1)
                    tmpnode.child.append(tmpnode1)
                    tmpnode1.father = tmpnode
                    tmpnode1.child.append(tmpnode2)
                    tmpnode2.father = tmpnode1
                    unknown.namewithouttype = " ".join(tmpnode.printTreeWithoutType(tmpnode).split()[:-1])
                    fans.append(unknown.printTreeWithoutType(ans.solveroot))
        elif unknown.father.namewithouttype == 'member':
            classname = ''
            if unknown.father.namewithouttype == 'member':
                #if x.father.father.father and x.father.father.father.name == 'This':
                for x in unknown.father.father.child:
                    if x.namewithouttype == 'qualifier':
                        if x.child[0].namewithouttype in typedic: 
                            classname = typedic[x.child[0].namewithouttype]
                            break
                        else:
                            for f in classcontent[sclassname + '.java']['classes'][0]['fields']:
                                if f['name'] == x.child[0].namewithouttype[:-4]:
                                     classname = f['type']
                                     break
                        if x.child[0].namewithouttype[:-4] + ".java" in classcontent:
                            classname = x.child[0].namewithouttype[:-4]
            #print(0, classname, ans.type)
            if classname == '':
                classbody = classcontent[sclassname + '.java']['classes']
            elif classname != '':
                if classname + ".java" not in classcontent:
                    #print(5, classname )
                    hasclass = False
                    for nclass in classcontent[sclassname + '.java']['classes']:
                        if nclass['name'] == classname:
                            classbody = [nclass]
                            hasclass = True
                    if not hasclass:
                        return []
                classbody = classcontent[classname + '.java']['classes']
            vtype = ""
            #print('type', ans.type)
            if 'fields' in classbody[0]:
                for x in classbody[0]['fields']:
                    if x['name'] == ans.type[:-4]:
                        vtype = x['type']
                        break
            if unknown.father.father.father.father and (unknown.father.father.father.father.namewithouttype == 'MethodInvocation' or unknown.father.father.father.father.namewithouttype == 'ClassCreator') and ans.type == "":
                mname = ""
                tname = ""
                if unknown.father.father.father.father.namewithouttype == "MethodInvocation":
                    tname = 'member'
                else:
                    tname = 'type'
                for s in unknown.father.father.father.father.child:
                    if s.namewithouttype == 'member' and tname == 'member':
                        mname = s.child[0].namewithouttype
                    if s.namewithouttype == 'type' and tname == 'type':
                        mname = s.child[0].child[0].child[0].namewithouttype
                idx = unknown.father.father.father.child.index(unknown.father.father)
                #print(idx)
                if tname == 'member':
                    for f in classbody[0]['methods']:
                        if f['name'] == mname[:-4] and idx < len(f['params']):
                            vtype = f['params'][idx]['type']
                            print(vtype, f['name'])
                            break
                else:
                    if mname[:-4] + ".java" not in classcontent:
                        return []
                    for f in classcontent[mname[:-4] + ".java"]['classes'][0]['methods']:
                        #print(f['name'], f['params'], mname[:-4])
                        if f['name'] == mname[:-4] and idx < len(f['params']):
                            vtype = f['params'][idx]['type']
                            break
            if True:
                try:
                    if 'IfStatement' in ans.getTreestr() and unknown.father.father.father.father.father and unknown.father.father.father.father.father.namewithouttype == 'condition':
                        vtype = 'boolean'
                except:
                    pass
                try:
                    if 'IfStatement' in ans.getTreestr() and unknown.father.father.father.father.father.father.father and unknown.father.father.father.father.father.father.father.namewithouttype == 'condition':
                        vtype = 'boolean'
                except:
                    pass
                if 'null' in ans.getTreestr() and unknown.father.father.father.father.father and unknown.father.father.father.father.father.namewithouttype == 'condition':
                    vtype = 'object'
                if 'fields' in classbody[0]:
                    for x in classbody[0]['fields']:
                        #print(classname, x['type'], x['name'], vtype, ans.type)
                        if x['type'] == vtype or (x['type'] == 'double' and vtype == 'int') or vtype == "object":
                            if x['name'] in ans.solveroot.getTreestr():
                                continue
                            unknown.namewithouttype = x['name'] + "_ter"
                            fans.append(unknown.printTreeWithoutType(ans.solveroot))
    return fans
def extarctmode(root):
    mode = 0
    if len(root.child) == 0:
        return 0, None
    if root.child[0].name == 'modified':
        mode = 0
    elif root.child[0].name == 'add':
        mode = 1
    else:
        return 0, None
        print(root.printTree(root))
        #assert(0)
    root.child.pop(0)
    return mode, root

def solveone(data, model, savepath):#(treestr, prob, model, subroot, vardic, typedic, idx, idss, classname, mode):
    #os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"
    #assert(len(data) <= 40)
    outputLen = 30
    if len(data) < 60 and 'Closure' not in data[0]['idss']:
        outputLen = 60
    args.batch_size = 60
    args.CodeLen = 30
    dev_set = SumDataset(args, "test", True)
    '''tmpdata = []
    for i in range(len(dev_set.data)):
        tmpdata.append(dev_set.data[i][19])
    tmpnl = dev_set.nl'''
    dev_set.preProcessOne(data)#x = dev_set.preProcessOne(treestr, prob)
    '''rrdict = {}
    for x in dev_set.Nl_Voc:
        rrdict[dev_set.Nl_Voc[x]] = x
    for i in range(len(dev_set.data)):
        if i == 0:
            for j in range(len(dev_set.data[i][0])):
                if dev_set.data[i][0][j] != tmpdata[i][j]:
                    print(dev_set.data[i][0][j], rrdict[tmpdata[i][j]], dev_set.nl[0][0][j])
        if i != 0 and i != 1:
            print(i)
            print(dev_set.data[i][0], tmpdata[i + 6])'''
    
    #dev_set.nl = [treestr.split()]
    #dev_set.data[5][0] = tmpdata[5 + 6]
    indexs = 0
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0)
    savedata = []
    patch = {}
    for x in tqdm(devloader):
        if indexs < 0:
            indexs += 1
            continue
        #print(indexs,indexs * args.batch_size, data[5]['oldcode'])
        #print(x[0][0], dev_set.data[0][idx])
        #assert(np.array_equal(x[0][0], dev_set.datam[0][4]))
        #assert(np.array_equal(x[1][0], dev_set.datam[1][4].toarray()))
        #assert(np.array_equal(x[2][0], dev_set.datam[8][4]))
        #assert(np.array_equal(x[3][0], dev_set.datam[9][4]))
        #print(data[indexs]['mode'], data[indexs]['oldcode'])
        '''lprofiler = LineProfiler()
        lp_wrapper = lprofiler(BeamSearch)
        lp_wrapper((x[0], x[1], None, None, None, None, None, None, x[2], x[3], x[4], x[5]), dev_set, model, 150, args.batch_size, indexs)
        lprofiler.print_stats()'''
        #assert(0)
        ans = BeamSearch((x[0], x[1], None, None, None, None, None, None, x[2], x[3], x[4], x[5]), dev_set, model, 200, args.batch_size, indexs)
        for i in tqdm(range(len(ans))):
            currid = indexs * args.batch_size + i
            idss = data[currid]['idss']
            if data[currid]['extended'] and 'Closure' not in idss:
                outputLen = 200
            subroot = data[currid]['subroot']
            if os.path.exists("generation/bugs-QuixBugs/bugs/%s/names.json" % idss):
                classcontent = json.load(open("generation/bugs-QuixBugs/bugs/%s/names.json" % idss, 'r') )
            else:
                assert(0)
                classcontent = []
            classcontent.extend(json.load(open("generation/temp.json", 'r')))
            rrdicts = {}
            for x in classcontent:
                rrdicts[x['filename']] = x
                if 'package_name' in x:
                    rrdicts[x['package_name'] + "." + x['filename']] = x
            vardic = data[currid]['vardic']
            typedic = data[currid]['typedic']
            classname = data[currid]['classname']#data[currid]['classname'].split(".")[-1]
            #print(vardic)
            #assert(0)
            mode = data[currid]['mode']
            rrdict = {}
            for x in vardic:
                rrdict[vardic[x]] = x
            for j in (range(len(ans[i]))):
                if j > outputLen:
                    break
                mode, ans[i][j].root = extarctmode(ans[i][j].root)
                if ans[i][j].root is None:
                    continue
                #print(j, ans[i][j].root.printTree(ans[i][j].root), ans[i][j].prob)
                applyoperater(ans[i][j], subroot)
                #print(j, ans[i][j].root.printTree(ans[i][j].solveroot))
                an = replaceVar(ans[i][j].solveroot, rrdict)
                print(j, ans[i][j].root.printTree(ans[i][j].solveroot))
                if not an:
                    continue
                #print(7, ans[i][j].type)
                try:
                    if ans[i][j].solveroot:
                        ans[i][j].root.printTree(ans[i][j].solveroot)
                    tcodes = solveUnknown(ans[i][j], vardic, typedic, rrdicts, classname, mode)
                except Exception as e:
                    #traceback.print_exc()
                    tcodes = []
                if len(tcodes) > 20:
                    tcodes = tcodes[:20]
                #print(tcodes, subroot.printTreeWithoutTer(subroot))
                for code in tcodes:
                    #if code.split(" ")[0] != 'root':
                    #    assert(0)
                    #print(code.replace("_ter ", " ").strip().split()[1:], subroot.printTreeWithoutTer(subroot).split())
                    if code.replace("_ter ", " ").strip().split()[1:-1] == subroot.printTreeWithoutTer(subroot).split():
                        continue
                    if str(mode) + code + str(data[currid]['line']) not in patch:
                        patch[str(mode) + code + str(data[currid]['line'])] = 1
                    else:
                        continue
                    savedata.append({'id':currid, 'idss':idss, 'precode':data[currid]['precode'], 'aftercode':data[currid]['aftercode'], 'oldcode':data[currid]['oldcode'], 'filename':data[currid]['filepath'], 'mode':mode, 'code':code, 'line':data[currid]['line'], 'isa':data[currid]['isa'], 'index':len(savedata), 'prob':ans[i][j].prob})
        indexs += 1
        #for x in savedata:
        #    print(x['oldcode'], x['code'])
        #exit(0)
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)
    #print(x[0][0], dev_set.data[0][idx])
    #assert(np.array_equal(x[0][0], dev_set.data[0][idx]))
    #assert(np.array_equal(x[1][0], dev_set.data[1][idx].toarray()))
    #assert(np.array_equal(x[2][0], dev_set.data[8][idx]))
    #assert(np.array_equal(x[3][0], dev_set.data[9][idx]))
    open('%s/%s.json' % (savepath, data[0]['idss']), 'w').write(json.dumps(savedata, indent=4))
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    if sys.argv[1] == "train":
        train()
    if sys.argv[1] == 'pretrain':
        pretrain()
    else:
        profile = line_profiler.LineProfiler()
        profile.enable()
        test()
        profile.disable()
        profile.print_stats(sys.stdout)
     #test()




