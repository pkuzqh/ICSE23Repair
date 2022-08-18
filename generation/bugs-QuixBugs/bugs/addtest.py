lst = ['BREADTH_FIRST_SEARCH.java', 'DEPTH_FIRST_SEARCH.java', 'DETECT_CYCLE.java', 'KTH.java', 'TOPOLOGICAL_ORDERING.java']
import os
for x in lst:
    os.system('cp -r BITCOUNT.java %s'%x)
    os.system('cp ../../QuixBugs-master/java_programs/%s %s/src/java_programs/'%(x, x))
