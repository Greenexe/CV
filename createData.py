
import os
import math
import numpy as np

path = './/archive//ICUDatasetProcessed'

files = []




# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))


preprocesedData=[]
for f in files:
    with open(f) as dataClass:
       preprocesedData.append(dataClass.readlines())


temp=[]
for f in preprocesedData:
    for line in f[1:]:
        temp.append(line.replace('\n','').split(','))

readyData=[]

for l in temp:
    if type(l)==list:
        for i in range(len(l)):
            if i==40 or i==44:
                l[i]='-'
            if ':' in l[i]:
                l[i]=  str(len(l[i].split(":")))
            if 'e' in l[i] and (i==0 or i==8):
                l[i]=str(float(l[i].split('e')[0])*pow(10,float(l[i].split('e')[1])))
            if i==3 or i==4:
                octets = l[i].split('.')
                count = 0
                for n, octet in enumerate(octets):
                    count += int(octet) << 8*(len(octets)-(n+1))
                l[i]=str(count)

            if 'x' in l[i] and (i!=40 or i!=44):
                l[i]=str(int(l[i],base=16))





        if True:
            if l[-2]=='Attack':
                rest=[1,0]
            elif l[-2]=='environmentMonitoring':
                rest=[0,1]
            elif l[-2]=='patientMonitoring':
                rest=[0,1]


            l=l[:-2]
            t=[]
            for x in l:
                if x!='-':
                    try:
                        t.append(float(x))
                    except:
                        t.append(0.0)
            readyData.append([t,rest])
            

        else:   
            if l[-2]=='Attack':
                rest=[1,0,0]
            elif l[-2]=='environmentMonitoring':
                rest=[0,1,0]
            elif l[-2]=='patientMonitoring':
                rest=[0,0,1]

            l=l[:-2]
            t=[]
            for x in l:
               if x!='-':
                    try:
                        t.append(float(x))
                    except:
                        t.append(0.0)
            readyData.append([t,rest])






#normalization minmax

minValue=[]
maxValue=[]


for x in readyData[0][0]:
    minValue.append(math.inf)
    maxValue.append(-math.inf)

for i in range(len(readyData)):
    for j in range(len(readyData[i][0])):
        x = readyData[i][0][j]
        if x<minValue[j]:
            minValue[j]=x
        
        elif x>maxValue[j]:
            maxValue[j]=x



noneChange=[]
for x in range(len(minValue)):
    if minValue[x]==maxValue[x]:
        noneChange.append(x)


tabX=[]
tabY=[]

for i in range(len(readyData)):
    newX=[]
    for j in range(len(readyData[i][0])):
        if j not in noneChange:
            if (readyData[i][0][j]-minValue[j])/(maxValue[j]-minValue[j]) !=  0:
                newX.append(round(np.log((readyData[i][0][j]-minValue[j])/(maxValue[j]-minValue[j])),8))
            else:
                newX.append(round((readyData[i][0][j]-minValue[j])/(maxValue[j]-minValue[j]),8))
    tabX.append(newX)
    
    if True:
       tabY.append(readyData[i][1][0])
    else:   
        tabY.append(readyData[i][1][0])
    

tabX=np.array(tabX)
tabY=np.array(tabY)
#tabX = np.array2string(tabX, suppress_small = True)
#tabY = np.array2string(tabY, suppress_small = True)
#np.savez_compressed('IoT_tabX_2.npz',tabX)
#np.savez_compressed('IoT_tabY_2.npz',tabY)
#print(np.shape(tabX))

np.savetxt("IoT_tabX_2.csv", tabX, delimiter=",",fmt='%f')
np.savetxt("IoT_tabY_2.csv", tabY, delimiter=",")



