

import numpy as np

words=[]
tags=[]

# Reading training data set                           
for line in (open('en_train.txt').readlines()):
	if not line.strip():
		continue	
	d=line.split()
	words.append(d[0]);        
	tags.append(d[1]);

# List of unique words and tags
words_unique=[]
tags_unique=[]
for i,j in zip(words,tags):
	if not i in words_unique:
		words_unique.append(i)
	if not j in tags_unique:
		tags_unique.append(j)

# Computing A,B,pi
v=len(tags_unique)
h=len(words_unique)
B=np.zeros((v,h))
A=np.zeros((v,v))
pi=np.zeros((1,v))
l_tag = '.'
for tag,word in zip(tags,words):
	B[tags_unique.index(tag),words_unique.index(word)]+=1
	if l_tag == '.':
		pi[tags_unique.index(tag)] += 1
	else:
		A[tags_unique.index(l_tag),tags_unique.index(tag)] += 1
	l_tag = tag
pi=pi/np.sum(pi)
for V in range(0,v):
	z1=0
	z2=0
	for V1 in range(0,v-1):
		z1=z1+A[V,V1]
	for V1 in range(0,h-1):
		z2=z2+B[V,V1]	
	A[V,:]=(A[V,:]*1.0)/z1;
	B[V,:]=(B[V,:]*1.0)/z2;

# List of observed sequence
observed_words=[]
c=0                           
for line in (open('en_test1.txt').readlines()):
	if not line.strip():
		continue	
	p=line.split()
	c=c+1
	if c==10:
		break
	observed_words.append(p[0]);

# Viterbi implementation

# Initialization
l=len(observed_words)
alpha=np.zeros((v,l))
backtrack=np.zeros((v,l))
pi1=pi
for i in range(0,v):
    pi=np.append(pi,pi1,axis=0)
w=words_unique.index(observed_words[0])
for g in range(0,v):
	alpha[g,0]=np.dot(pi[g,g],B[g,w])
backtrack[:,0]=0

# Recursion
for i in range(1,l):                  
    for j in range(0,v):
		t1=0
		t3=0
		r1=[]
		for j1 in range(0,v):            
			t=max(t1,(alpha[j1,i-1]*A[j1,j]*B[j,words_unique.index(observed_words[i])]))
			t1=t
			t2=max(t3,alpha[j1,i-1]*A[j1,j])
			t3=t2
			r1.append(t3)
		backtrack[j,i]=np.argmax(r1)        
		alpha[j,i]=t1

# Backtracking
k1=0
k3=0
q2=[]
for k in range(0,v):                
    p=max(alpha[k,l-1],k1)
    k1=p
    q2.append(p)
q=np.argmax(q2)
y=[q]
for m in range(l-1,0,-1):          
    q4=backtrack[q,m]    
    q=q4
    y.append(q)    
for e in range(8,-1,-1):
   print tags_unique[int(y[e])],observed_words[8-e]















