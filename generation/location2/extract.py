import os
for i in range(134, 177):
    #os.system("mkdir Closure2/Closure_%d"%i)
    #os.system("cp Closure/%d/parsed_ochiai_result , i))
    s = open("Closure2/Closure_%d/Ochiai"%i, "r")
    f = open("Closure2/Closure_%d/Ochiai.txt"%i, "w")
    for x in s:
        f.write("@".join(x.strip().split()[0].split("#")) + "\n")
    
