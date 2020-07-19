import numpy as np
import sys
import os
from Regression import *
import subprocess

class test_boxpushing:
	def __init__(self,sensitivity):
		self.sensitivity = sensitivity
		if sensitivity =="unavoidable":
			self.testprob = ['grid-3_unavoidable','grid-3-t1_unavoidable','grid-3-t2_unavoidable','grid-3-t3_unavoidable','grid-3-t5_unavoidable','grid-3-t7_unavoidable']

		else:
			self.testprob = ['grid-3','grid-3-t1','grid-3-t2','grid-3-t3','grid-3-t6','grid-3-t7']
			

	def getTrainProb(self):
		return self.testprob[0]

	def gettestProb(self):
		return self.testprob

	def getTestInstances(self):
		instances = []
		for prob in self.testprob:
			instances.append('--box=data/boxpushing/'+prob+".bp")
		return instances

	def getStates(self,filename):
		sa = []
		states= []
		bpfile=open(filename,"r")
		for line in bpfile:
			templine=line.rstrip()
			temp=templine.split(" ")
			sa.append(templine)
			index = templine.find(')') 
			states.append(templine[:index+1])

		bpfile.close()
		return sa,states

	def getTrainingFile(self,p):
		filename = "data/boxpushing/"+self.testprob[p]+"_Samples.txt"
		return filename

	def getRLTrainingFile(self,p):
		filename = "data/boxpushing/"+self.testprob[p]+"_RLSamples.txt"
		return filename

	def getTestingFile(self,p):
		filename = "data/boxpushing/"+self.testprob[p]+"_Testing.txt"
		return filename

	def generate_proactive_samples(self,tprob,costF, instanceId,slack):
		p =  instanceId
		slackip = '--slack='+str(slack)
		# generate testfile and processed file
		if(costF == "Demo-AM"):
			call(['./testNSE.out',tprob,'--gamma=0.95',slackip,'--algorithm=LLAO','--numObj=2','--v=1','--demo']) #training File
			trFile = 'data/boxpushing/'+ self.testprob[p] + "_Samples.txt"
			testingFile = 'data/boxpushing/'+ self.testprob[p] + "_Testing.txt"
			processedFile = 'data/boxpushing/'+ self.testprob[p] + "_Processed.csv"
			return trFile, testingFile, processedFile
		else:
			call(['./testNSE.out',tprob,'--gamma=0.95',slackip,'--algorithm=LLAO','--numObj=2','--v=10'])
			testingFile = 'data/boxpushing/'+ self.testprob[p] + "_Testing.txt"
			processedFile = 'data/boxpushing/'+ self.testprob[p] + "_Processed.csv"
			return testingFile, processedFile

	def generate_demo_samples(self,tprob, instanceId,slack,trials):
		p =  instanceId
		slackip = '--slack='+str(slack)
		# generate testfile and processed file
		db = '--trials='+str(trials)
		call(['./testNSE.out',tprob,'--gamma=0.95',slackip,'--algorithm=LLAO','--numObj=2','--v=1','--demo',db]) #training File
		trFile = self.getTrainingFile(p)
		testingFile = self.getTestingFile(p)
		processedFile = 'data/boxpushing/'+ self.testprob[p] + "_Processed.csv"
		return trFile, testingFile, processedFile
		# return trFile

	def generate_corrections(self,tprob,instanceId,trials,slack):
		p =  instanceId
		db = '--trials='+str(trials)
		slackip = "--slack="+str(slack)
		# generate testfile and processed file
		call(['./testNSE.out',tprob,'--gamma=0.95',slackip,'--algorithm=LLAO','--numObj=2','--v=201']) #policy File
		call(['./testNSE.out',tprob,'--gamma=0.95','--slack=0','--algorithm=LLAO','--numObj=2','--v=1','--corrections',db]) #training File -- agent trajectory
		trFile = self.getTrainingFile(p)
		testingFile = self.getTestingFile(p)
		policyFile = 'data/boxpushing/' +self.testprob[p] + "_Policy.txt"

		#read policy file and store (s,a) for corrections.
		sa = []
		states= []
		bpfile=open(policyFile,"r")
		for line in bpfile:
			templine=line.rstrip()
			templine=templine.replace(', ','')
			temp=templine.split(" ")
			sa.append(temp[0] + " "+temp[3]) #states and states, action hvalue
			states.append(temp[0])
		bpfile.close()
		#read agent trajectory
		traj_states= []
		traj_sa = []
		to_write = ""
		bpfile=open(trFile,"r")
		for line in bpfile:
			templine=line.rstrip()
			templine=templine.replace(', ','')
			temp=templine.split(" ")
			traj_states.append(temp[0])
			traj_sa.append(temp[0]+" "+temp[1])
		bpfile.close()

		bpfile = open(testingFile,'r')
		for line in bpfile:
			templine=line
			templine=templine.rstrip().replace(', ','')
			temp=templine.split(" ")
			curr_sa = temp[0] + ' ' + temp[3]
			if(curr_sa in traj_sa and curr_sa in sa):
				to_write += line[:-2] + '0\n'
			elif (temp[0] in traj_states and curr_sa in traj_sa and curr_sa not in sa): 
				to_write += line[:-2] + '10\n'
			elif temp[0] not in traj_states: #state not found
				to_write += line[:-2] + '0\n'

		bpfile.close()
		bpfile = open(trFile,"w")
		bpfile.write(to_write)
		bpfile.close()		
		processedFile = 'data/boxpushing/'+ self.testprob[p] + "_Processed.csv"
		return trFile, testingFile, processedFile

	def generate_reactive_samples(self,tprob,epsilon, episodes,slack, instanceId):
		eps = '--epsilon='+str(epsilon)
		epsiode_val = '--episodes='+str(episodes)
		slackip = '--slack='+str(slack)
		p = instanceId
		# Generate training with best episodes found
		call(['./testNSE.out',tprob,'--gamma=0.95',slackip,'--algorithm=Exploration','--numObj=2','--v=1',eps,epsiode_val]) #Training samples
		trainFile = self.getRLTrainingFile(p)
		testingFile = self.getTestingFile(p)
		processedFile = 'data/boxpushing/'+ self.testprob[p] + "_RLProcessed.csv"
		return trainFile, testingFile, processedFile

	def solve_learnedReward(self,tprob,probName, processedFile, costF, slack):
		rewardFile= '--testing='+processedFile
		slackip = '--slack='+str(slack)
		cf = '--cost='+costF
		#Solve the problem with the learned reward, best slack
		opline=subprocess.check_output(['./testNSE.out',tprob,'--gamma=0.95',slackip,'--algorithm=OF','--numObj=3','--v=100',rewardFile,cf])
		return opline

			
	def ParseFiles(self,filename):
		x=[]
		y=[]
		x_full=[]
		bpfile=open(filename,"r")
		for line in bpfile:
			templine=line
			templine=templine.replace('(','').replace(')','').rstrip()
			templine=templine.replace(',','')
			temp=templine.split(" ")
			state=[]
			state.append(int(temp[0]))
			state.append(int(temp[1]))
			state.append(int(temp[2]))
			state.append(int(temp[3]))
			state.append(int(temp[4]))
			state.append(int(temp[7]))
			x_full.append(state)

			state_reg=[] #factors used for testing
			state_reg.append(int(temp[2]))
			state_reg.append(int(temp[3]))
			state_reg.append(int(temp[4]))
			x.append(state_reg)
			y.append(float(temp[8]))

		bpfile.close()
		return x,y,x_full