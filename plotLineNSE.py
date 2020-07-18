##########################################################################
# Author: Sandhya Saisubramanian, UMass Amherst
# Execution: python plotLineNSE [logs] [domain_name]
# Example: "python plotLineNSE Boxpushing-Logs.csv bp" will
#  			generate plots for boxpushing domain and save plots as 
# 			bp_HA.png and bp_RL.png 
#########################################################################


import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


filename = sys.argv[1]
domain_name  = sys.argv[2]


baseline_techniques=['IF','UB']
if("unavoidable" in filename):
	baseline_techniques=['Opt','UB']

RL_techniques = ['conservative','moderate','radical']
HA_techniques = ['RR','HA-L','HA-S','Demo-AM','correction','deviation']
HA_legend = ['Random query','HA-L','HA-S','Demo-AM','Correction','RR']
RL_legend = ['Conservative','Moderate','Radical']
baseline_legend = ['Oracle', 'No queries']
R1_techniques = ['NLLAO','UB','Opt']


bcolor = ['black','magenta','teal', 'slategrey','lavender','lightblue']

colors = ['blue','red','green','darkorange']
ecolor = ['cyan','salmon','lightgreen','peachpuff']
   

testprob = []
HA_budget = ['100','200','500','1000','2000','4000','7000']
RL_budget = ['10','25','50','75','100']


def load_problems():
	myfile = open(filename,"r")
	index = -1
	for line in myfile:
		index += 1
		if(index == 0):
			continue

		temp = line.split(",")
		if(temp[0].strip() in testprob):
			continue
		else:
			testprob.append(temp[0].strip())
	myfile.close()

load_problems()

total_techniques_count  = len(baseline_techniques) + len(RL_techniques) + len(HA_techniques)
baseline_R2 = np.zeros((len(baseline_techniques),len(testprob)))
baseline_R2_mean = np.zeros((len(baseline_techniques)))
baseline_R2_err = np.zeros((len(baseline_techniques)))
HA_R2 = np.zeros((len(HA_budget),len(HA_techniques), len(testprob)))
RL_R2  = np.zeros((len(RL_budget),len(RL_techniques), len(testprob)))
HA_R2_mean = np.zeros((len(HA_budget),len(HA_techniques)))
RL_R2_mean = np.zeros((len(RL_budget),len(RL_techniques)))
HA_R2_err = np.zeros((len(HA_budget),len(HA_techniques)))
RL_R2_err = np.zeros((len(RL_budget),len(RL_techniques)))


R1 = np.zeros((len(testprob),len(R1_techniques)))
R1_err = np.zeros((len(testprob),len(R1_techniques)))
R1_mean = np.zeros((len(R1_techniques)))
R1_err_mean = np.zeros((len(R1_techniques)))

myfile = open(filename,"r")
index = -1
for line in myfile:
	index += 1
	if(index == 0):
		continue

	temp = line.split(",")

	p = testprob.index(temp[0].strip())
	if(temp[2].strip() in baseline_techniques):
		t = baseline_techniques.index(temp[2].strip())
		baseline_R2[t][p] = float(temp[5])
		baseline_R2_err[t] += float(temp[6])/10.0

	elif(temp[2].strip() in HA_techniques):
		t = HA_techniques.index(temp[2].strip())

		if temp[1].strip() in HA_budget:
			b = HA_budget.index(temp[1].strip())
			HA_R2[b][t][p]  = float(temp[5])
			HA_R2_err[b][t] += float(temp[6])/10.0

	elif(temp[2].strip() in RL_techniques):
		t = RL_techniques.index(temp[2].strip())
		if temp[1].strip() in RL_budget:
			b = RL_budget.index(temp[1].strip())
			RL_R2[b][t][p]  = float(temp[5])
			RL_R2_err[b][t] += float(temp[6])/10.0

	if(temp[2].strip() in R1_techniques):
		t = R1_techniques.index(temp[2].strip())
		R1[p][t] = float(temp[3])
		R1_err[p][t] = float(temp[4])/10.0

myfile.close()

for t in range(len(baseline_techniques)):
	baseline_R2_mean[t] = np.sum(baseline_R2[t,:])/len(testprob)

for b in range(len(HA_budget)):
	for t in range(len(HA_techniques)):
		HA_R2_mean[b][t] = np.sum(HA_R2[b,t,:])/len(testprob)

HA_R2_err = HA_R2_err/len(testprob)

for b in range(len(RL_budget)):
	for t in range(len(RL_techniques)):
		RL_R2_mean[b][t] = np.sum(RL_R2[b,t,:])/len(testprob)

RL_R2_err = RL_R2_err/len(testprob)
baseline_R2_err = baseline_R2_err/len(testprob)

R1_mean = R1/len(testprob)
R1_err_mean = R1_err/len(testprob)


N = np.arange(len(HA_budget))

linestyles = ['-', '--', '-.',':']

fig = plt.figure()
ax = fig.add_subplot(111)
for t in range(len(HA_techniques)):
	if(HA_techniques[t] == "RR"):
		ax.plot(HA_R2_mean[:,t], color = 'brown',label=HA_legend[t],linewidth=2, linestyle = '-')
		ax.fill_between(N,HA_R2_mean[:,t]+ HA_R2_err[:,t], HA_R2_mean[:,t]- HA_R2_err[:,t], color='beige')
	elif(HA_techniques[t] == "deviation"):
		ax.plot(HA_R2_mean[:,t], color = 'teal',label=HA_legend[t],linewidth=2, marker="^")
		ax.fill_between(N,HA_R2_mean[:,t]+ HA_R2_err[:,t], HA_R2_mean[:,t]- HA_R2_err[:,t], color='lightblue')
	else:
		ax.plot(HA_R2_mean[:,t], color = colors[t-1],label=HA_legend[t],linewidth=2, linestyle = linestyles[t-1])
		ax.fill_between(N,HA_R2_mean[:,t]+ HA_R2_err[:,t], HA_R2_mean[:,t]- HA_R2_err[:,t], color=ecolor[t-1])
for t in range(len(baseline_techniques)):
	temp = np.empty((len(HA_budget)))
	temp.fill(baseline_R2_mean[t])
	for b in range(len(HA_budget)):
		ax.plot(temp, color = bcolor[t], linewidth =2, label= baseline_legend[t],marker = "^")
		ax.fill_between(N,baseline_R2_mean[t]+ baseline_R2_err[t], baseline_R2_mean[t]- baseline_R2_err[t], color=bcolor[t+3])
ax.set_xticklabels(HA_budget,fontsize='16')
ax.set_xticks(N)
x1,x2,y1,y2 = plt.axis()
ax.set_ylabel('Penalty',fontsize='16')
ax.set_xlabel('Budget',fontsize='16')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
box = ax.get_position()
ax.legend(by_label.values(), by_label.keys(),loc ='upper left',ncol=4,handletextpad=0.1,columnspacing=0.8,bbox_to_anchor=(-0.08, box.height+0.35),fancybox=True)
plt.savefig(domain_name+'_HA.png',bbox_inches='tight')
plt.clf()





N2 = np.arange(len(RL_budget))
fig = plt.figure()
ax = fig.add_subplot(111)
for t in range(len(RL_techniques)):
	ax.plot(RL_R2_mean[:,t], color = colors[t],label= RL_legend[t],linewidth=3,linestyle = linestyles[t])
	ax.fill_between(N2,RL_R2_mean[:,t]+ RL_R2_err[:,t], RL_R2_mean[:,t]- RL_R2_err[:,t], color=ecolor[t])
for t in range(len(baseline_techniques)):
	temp = np.empty((len(RL_budget)))
	temp.fill(baseline_R2_mean[t])
	for b in range(len(RL_budget)):
		ax.plot(temp, color = bcolor[t], linewidth =2, label= baseline_legend[t],marker = "^")
		ax.fill_between(N2,baseline_R2_mean[t]+ baseline_R2_err[t], baseline_R2_mean[t]- baseline_R2_err[t], color=bcolor[t+3])

t= 0
temp = np.empty((len(RL_budget)))
s = len(HA_budget)-1
temp.fill(HA_R2_mean[s][0])
# Plot RR
ax.plot(temp, color = 'brown', linewidth =2, label=HA_legend[0], linestyle = '-')
ax.fill_between(N2,HA_R2_mean[s][t]+ HA_R2_err[s][t], HA_R2_mean[s][t] - HA_R2_err[s][t], color='beige')

# Plot Deviation
t = len(HA_techniques)-1
temp = np.empty((len(RL_budget)))
s = len(HA_budget)-1
temp.fill(HA_R2_mean[s][t])
ax.plot(temp, color = 'teal',label=HA_legend[t],linewidth=2, marker="^")
ax.fill_between(N2,HA_R2_mean[s][t]+ HA_R2_err[s][t], HA_R2_mean[s][t]- HA_R2_err[s][t], color='lightblue')

ax.set_xticklabels(RL_budget,fontsize='16')
ax.set_ylabel('Penalty',fontsize='16')
ax.set_xlabel('Budget',fontsize='16')
ax.set_xticks(N2)
x1,x2,y1,y2 = plt.axis()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
box = ax.get_position()
ax.legend(by_label.values(), by_label.keys(),loc ='upper left',ncol=4,handletextpad=0.1,columnspacing=0.8,bbox_to_anchor=(0, box.height+0.35),fancybox=True)
plt.savefig(domain_name+'_RL.png',bbox_inches='tight')
plt.clf()

