def parse(f_sp,f_mat):
    l_sp=[l for l in f_sp.read().splitlines() if l]
    tests=[l.split(' ') for l in f_mat.read().splitlines() if l]
    
    ochiais=[]
    for ind,sp in enumerate(l_sp):
        ef=ep=nf=np=0
        for test in tests:
            res=test[-1]
            execute={'1':True, '0':False}[test[ind]]
            if res=='+': # pass
                if execute:
                    ep+=1
                else:
                    np+=1
            elif res=='-': # fail
                if execute:
                    ef+=1
                else:
                    nf+=1
            else:
                1/0
        if ef:
            ochiais.append([sp,ef/(((ef+nf)*(ef+ep))**.5)])
        else:
            ochiais.append([sp,0])
    ochiais.sort(key=lambda x:-x[1])
    return ochiais

import os
for proj in os.listdir('.'):
    if proj in ['logs','parsefiles.py', 'extract.py']:
        continue
    for ver in os.listdir(proj):
        d='%s/%s/'%(proj,ver)
        print('==',proj,ver)
        os.system('rm %sspectra'%d)
        os.system('rm %smatrix'%d)
        os.system('rm %stests'%d)
        os.system('rm %s.spectra'%d)
        os.system('rm %sgzoltar.ser'%d)
        os.system('rm -rf %s54'%d)

                #p=parse(f_sp,f_mat)
                #with open(d+'parsed_ochiai_result','w') as f:
                #    for l,prob in p:
                #        f.write('%s\t%s\n'%(l,prob))
