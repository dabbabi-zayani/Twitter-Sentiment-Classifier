# preparing 3cixty reviews
f=open('../data/3cixty/reviews.csv','r')
fo=open('../data/3cixty/3cixty.csv','w')
line=f.readline()
line=f.readline()
while (line):
   li=line.split('\t')
   l=li[1][1:-2]
   l=l.replace("&quot","")
   print l
   fo.write(l+'\n')
   line=f.readline()

f.close()
fo.close()
