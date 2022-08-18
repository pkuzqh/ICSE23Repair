import json
import sys
import os
from Searchnode import Node
from stringfycode import stringfyRoot
import javalang
import subprocess
import time
import signal
import traceback
import difflib
from tqdm import tqdm
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-1', 'Closure-10', 'Closure-14', 'Closure-15', 'Closure-18', 'Closure-31', 'Closure-33', 'Closure-38', 'Closure-51', 'Closure-62', 'Closure-63', 'Closure-70', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-107', 'Closure-118', 'Closure-113', 'Closure-124', 'Closure-125', 'Closure-129', 'Lang-6', 'Lang-16', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-38', 'Lang-39', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-5', 'Math-25', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-57', 'Math-58', 'Math-59', 'Math-69', 'Math-70', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Lang-43', 'Math-50', 'Math-98', 'Time-7', 'Mockito-38', 'Mockito-22', 'Mockito-29', 'Mockito-34', 'Closure-104', 'Math-27']
lst = ['Lang-39', 'Lang-63', 'Math-88', 'Math-82', 'Math-20', 'Math-28', 'Math-6', 'Math-72', 'Math-79', 'Math-8', 'Math-98']#['Closure-38', 'Closure-123', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
bugid = sys.argv[1]
def convert_time_to_str(time):
    #时间数字转化成字符串，不够10的前面补个0
    if (time < 10):
        time = '0' + str(time)
    else:
        time=str(time)
    return time

def sec_to_data(y):
    h=int(y//3600 % 24)
    d = int(y // 86400)
    m =int((y % 3600) // 60)
    s = round(y % 60,2)
    h=convert_time_to_str(h)
    m=convert_time_to_str(m)
    s=convert_time_to_str(s)
    d=convert_time_to_str(d)
    # 天 小时 分钟 秒
    return d + ":" + h + ":" + m + ":" + s
def getroottree2(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for x in tokens[1:]:
        if x != "^":
            nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root
import psutil
def kill_tree(process):
    p = psutil.Process(process)
    ps = p.children(recursive=True)
    ps.append(p)
    for proc in ps:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
lst = [bugid]
starttime = time.perf_counter()
timelst = []
for x in lst:
    if x != bugid:
        continue
    #wf = open('patches/' + x + "patch.txt", 'w')
    patches = json.load(open("patch2/%s.json"%x, 'r'))
    curride = ""
    proj = x.split("-")[0]
    bid = x.split("-")[1]
    x = x.replace("-", "")
    if os.path.exists('buggy%s' % x):
        subprocess.call('rm -rf buggy%s' % x, stderr=subprocess.DEVNULL, shell=True)
        #os.system('rm -rf buggy%s' % x)
    if not os.path.exists('nochange%s'%x):
        subprocess.call('defects4j checkout -p %s -v %s -w nochange%s' % (proj, bid + 'b', x), stderr=subprocess.DEVNULL, shell=True)
        #os.system('defects4j checkout -p %s -v %s -w nochange%s' % (proj, bid + 'b', x))
    subprocess.call("defects4j checkout -p %s -v %s -w buggy%s" % (proj, bid + 'b', x), stderr=subprocess.DEVNULL, shell=True)
    #os.system("defects4j checkout -p %s -v %s -w buggy%s" % (proj, bid + 'b', x))
    xsss = x
    testmethods = subprocess.Popen('defects4j export -w buggy%s -p tests.trigger'%x, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, shell=True).stdout.readlines()
    for i, p in enumerate(tqdm(patches, desc=bugid, ncols=80)):
        #print(time.perf_counter() - starttime)
        if time.perf_counter() - starttime > 18000:
            if os.path.exists('nochange%s' % x):
                os.system('rm -rf nochange%s' % x)
            if os.path.exists('buggy%s' % x):
                os.system('rm -rf buggy%s' % x)
            open('haseval2.txt', 'a').write(str(bugid) + '\n')
            exit(0)
        #if i != 18:
        #    continue
        endtime = time.time()
        #if endtime - starttime > 18000:
        #    open('timeg.txt', 'a').write(xsss + "\t" + sec_to_data(endtime - starttime) + "\n")
        #    exit(0)
        iden = x + str(p['line']) + p['filename'].replace("/", "")[::-1]
        if iden != curride:
            if curride != "":
                os.system('rm -rf buggy%s'%x)
        subprocess.call('defects4j checkout -p %s -v %s -w buggy%s' % (proj, bid + 'b', x), stderr=subprocess.DEVNULL, shell=True)
        #os.system('defects4j checkout -p %s -v %s -w buggy%s' % (proj, bid + 'b', x))
        curride = iden
        try:
            root = getroottree2(p['code'].split())
        except:
            #assert(0)
            continue
        if p['line'] == 178:
            continue
        mode = p['mode']
        filename = p['filename']
        filename = filename.replace("buggy%s/"%x, "nochange%s/"%x)
        totalcode = open(filename, 'r').read()
        lines = totalcode.split('\n')
        precode =  "\n".join(lines[p['precode'][0]:p['precode'][1]])
        aftercode = "\n".join(lines[p['aftercode'][0]:p['aftercode'][1]])
        oldcode = "\n".join(lines[p['oldcode'][0]:p['oldcode'][1]]) + '\n'
        #print(oldcode)
        if '-1' in oldcode and mode == 0:
            continue
        if mode == 1:
            aftercode = oldcode + aftercode
        lines = aftercode.splitlines()
        if 'throw' in lines[0] and mode == 1:
            for s, l in enumerate(lines):
                if 'throw' in l or l.strip() == "}":
                    precode += l + "\n"
                else:
                    break
            aftercode = "\n".join(lines[s:])
        if lines[0].strip() == '}' and mode == 1:
            precode += lines[0] + "\n"
            aftercode = "\n".join(lines[1:])
        #print(aftercode.splitlines()[:10])

        try:
            code = stringfyRoot(root, False, mode)
        except:
            #print(traceback.print_exc())
            continue
        if '-1' in code:
            continue
        if '<string>' in code:
            if '\'.\'' in oldcode:
                code = code.replace("<string>", '"."')
            elif '\'-\'' in oldcode:
                code = code.replace("<string>", '\'-\'')
            elif '\"class\"' in oldcode:
                code = code.replace("<string>", '"class"')
            else:
                code = code.replace("<string>", "\"null\"")
        if len(root.child) > 0 and root.child[0].name == 'condition' and mode == 0:
            if 'while' in oldcode:
                code = 'while' + code + "{"
            elif '} else if' in oldcode:
                code = '}else if' + code + "{"
            elif 'else if' in oldcode:
                code = 'else if' + code + "{"
            
            else:
                code = 'if' + code + "{"
        if code == "" and 'for' in oldcode and mode == 0 and '0' not in oldcode:
            code = oldcode + "if(0!=1)break;"
        if code == "" and 'if' in oldcode and mode == 0:
            code = 'if(false) {'
        if 'return _currentSegment;' in oldcode and 'if' in code and '&&' in code:
            code += 'return _currentSegment;}'
        if code == "" and 'if' in precode.splitlines()[-1] and '0x' in precode.splitlines()[-1]:
            precode = "\n".join(precode.splitlines()[:-1] + ['if(false){\n'])
        #special case
        if 'return' == code.strip()[:6] and mode == 1:
            code = "if(0!=1){" + code + "}"
        filepath2 = p['filename']#p['filename'].replace(proj + '/', x + '/')#'buggy%s'%x + p['filename'].replace("")
        lnum = 0
        for l in code.splitlines():
            if l.strip() != "":
                lnum += 1
            else:
                continue
        patched = False
        if mode == 1 and len(precode.splitlines()) > 0 and 'case' in precode.splitlines()[-1]:
            lines = precode.splitlines()
            for i in range(len(lines) - 2, 0, -1):
                if lines[i].strip() == '}':
                    break
            if len(lines) -2 - i < 4: 
                precode = "\n".join(lines[:i])
                aftercode = "\n".join(lines[i:]) + "\n" + aftercode
                patched = True
        if not patched and lnum == 1 and 'if' in code and mode == 1 and "0!=1" not in code:
            if p['isa'] or 'Child' in oldcode:
                code = code.replace("if", 'while')
            #print('ppp', precode.splitlines()[-1])
            if len(precode.splitlines()) > 0 and 'for' in precode.splitlines()[-1]:
                code = code + 'continue;\n}\n'    
            else:
                afterlines = aftercode.splitlines()
                lnum = 0
                rnum = 0
                ps = p
                for p, y in enumerate(afterlines):
                    if ps['isa'] and y.strip() != '':
                        aftercode = "\n".join(afterlines[:p + 1] + ['}'] + afterlines[p + 1:])
                        break
                    if '{' in y:
                        lnum += 1
                    if '{' not in  y  and p - 1 >= 0 and '{' not in afterlines[p - 1]:
                        aftercode = "\n".join(afterlines[:p] + ['}'] + afterlines[p:])
                        break
                    if '}' in y:
                        if lnum == 0:
                            aftercode = "\n".join(afterlines[:p] + ['}'] + afterlines[p:])
                            #assert(0)
                            break
                        lnum -= 1
            #print(code)
            tmpcode = precode + "\n" + code + '\n' + aftercode
            tokens = javalang.tokenizer.tokenize(tmpcode)
            parser = javalang.parser.Parser(tokens)
        else:
            if not patched:
                #print(1, code, oldcode)
                tmpcode = precode + "\n" + code + '\n' + aftercode
                tokens = javalang.tokenizer.tokenize(tmpcode)
                parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse()
        except:
            #print(code, p['code'])
            #assert(0)
            #print('ttttt')
            continue
        #print(filepath2)
        open(filepath2, "w").write(tmpcode)
        bugg = False
        for t in testmethods:
            t = t.decode('utf-8')
            #print(t.strip())
            cmd = 'defects4j test -w buggy%s/ -t %s' % (x, t.strip())
            Returncode = ""
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, start_new_session=True)
            while_begin = time.perf_counter()#time.time() 
            while True:                
                Flag = child.poll()
                #print(Flag)
                #print(time.time() - while_begin)
                if  Flag == 0:
                    Returncode = child.stdout.readlines()#child.stdout.read()
                    break
                elif Flag != 0 and Flag is not None:
                    bugg = True
                    break
                elif time.perf_counter() - while_begin > 15:
                    #print('ppp')
                    kill_tree(child.pid)#os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    bugg = True
                    break
                else:
                    time.sleep(1)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                continue
            elif len(log) == 0:
                bugg = True
                break
            else:
                #print(len(log), log[-1].decode('utf-8'))
                bugg = True
                break
        if not bugg:
            #print('s')
            cmd = 'defects4j test -w buggy%s/' % (x)
            Returncode = ""
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, start_new_session=True)
            while_begin = time.time() 
            while True:                
                Flag = child.poll()
                #print(time.time() - while_begin, Flag)
                if  Flag == 0:
                    Returncode = child.stdout.readlines()#child.stdout.read()
                    break
                elif Flag != 0 and Flag is not None:
                    bugg = True
                    break
                elif time.time() - while_begin > 180:
                    kill_tree(child.pid)#os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    bugg = True
                    break
                else:
                    time.sleep(1)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                #print('success')
                endtime = time.time()
                delta = difflib.unified_diff(totalcode.splitlines(), tmpcode.splitlines(), lineterm='\n')
                open("validation/patches-all/patches2/%spatch.txt"%bugid, 'w').write(filename + '\n' + "\n".join(delta))
                #print('\n'.join(delta))
                #assert(0)
                #open('timeg.txt', 'a').write(xsss + "\t" + sec_to_data(endtime - starttime) + "\n")
                #timelst.append(sec_to_data(endtime - starttime))
                #wf.write(curride + "\n")
                #wf.write("-" + oldcode + "\n")
                #wf.write("+" +  code + "\n")
                #wf.write("🚀\n")
                #wf.flush()    
                if os.path.exists('buggy%s' % x):
                    os.system('rm -rf buggy%s' % x)
                if os.path.exists('nochange%s' % x):
                    os.system('rm -rf nochange%s' % x)
                open('haseval2.txt', 'a').write(str(bugid) + '\n')
                exit(0)
        #exit(0)
    if os.path.exists('nochange%s' % x):
        os.system('rm -rf nochange%s' % x)
    if os.path.exists('buggy%s' % x):
        os.system('rm -rf buggy%s' % x)
    open('haseval2.txt', 'a').write(str(bugid) + '\n')
    endtime = time.time()

