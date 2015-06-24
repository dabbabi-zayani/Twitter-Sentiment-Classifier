# preparing 3cixty reviews
f=open('../data/3cixty/sparql','r')
fo=open('../data/3cixty/3cixty.txt','w')
line=f.readline()
line=f.readline()
while (line):
   li=line.split('\t')
   fo.write(li[1]+'\n')
   line=f.readline()

f.close()
fo.close()
