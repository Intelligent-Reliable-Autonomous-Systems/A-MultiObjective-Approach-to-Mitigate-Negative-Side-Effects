#######################################################################
# Author: Sandhya Saisubramanian
# Description: Code base for IJCAI'20 paper 
#		"A Multi-Objective Approach to Mitigate Negative Side Effects"
#######################################################################

############  	Execution:
# python executeNSE.py <domain_name> <sensitivity>
# domain_name: {"boxpushing","navigation"}
# sensitivity: {"","unavoidable"}
###########################################################################################

import numpy as np
import os
import subprocess
from Regression import *
from test_boxpushing import *
from test_navigation import *

domain_name = sys.argv[1]
sensitivity = ""
if(len(sys.argv) >2):
	sensitivity = sys.argv[2]
print("sensitivity=",sensitivity)



############################################################################
# Represents various feedback mechanisms for learning costs of NSE.
# RR: random query
# HA-L and HA-S: lenient and strict human approval.
# IF: impact free oracle -> completely avoids NSEs, when feasible.
# UB: No query case (executes primary policy)
# Opt: LLAO with perfect model of NSE and slack from Alg 1
# deviation: Implements "Penalizing side effects using stepwise relative reachability" 
# 			 with scalarization and modified deviation score.
############################################################################
techniques = ['IF','UB','RR', 'HA-L', 'HA-S' ,'Demo-AM','correction','conservative','moderate','radical','Opt','deviation']

HA_budget = ['100','200','500','1000','2000','4000','6000','7000']
RL_budget = ['10','25','50','75','100']


# Learning and Generalizing the learned reward:
def LearnReward(QueryBudget,ProcessedFile,testInput,x_test, y_test, x_train, y_train):
	rg = Regression()
	mse = rg.Predict(x_train,y_train,x_test,y_test,testInput, ProcessedFile,QueryBudget)
	return mse

# Finding the best slack for achieving 0 NSE. If 0 NSE is not possible, 50% opt is used as slack
def getSlack(problemFile):
	opline1 = subprocess.check_output(['./testNSE.out',problemFile,'--gamma=1.0','--algorithm=LLAO','--v=1','--numObj=1'])
	opline2 = subprocess.check_output(['./testNSE.out',problemFile,'--gamma=1.0','--algorithm=LLAO','--v=1','--numObj=1', '--impactFree'])

	v1 = float(opline1.strip())
	if sensitivity == "unavoidable":
		return  0.15 * v1
	
	v2 = float(opline2.strip())
	print(v1,v2)
	return abs(v2-v1)


def AvgReward(line):
	temp = line.split(" ")
	return temp[3],temp[5], temp[9],temp[11]

def readFiles(filename):
		sa = []
		states= []
		curr_file=open(filename,"r")
		for line in curr_file:
			templine=line.rstrip()
			templine=templine.replace(', ','')
			temp=templine.split(" ")
			sa.append(templine)
			states.append(temp[0])

		curr_file.close()
		return sa,states

def demoAcceptable(training, testing):
	demo_data, demo_states = readFiles(training)
	training_content=""
	curr_file=open(testing,"r")
	for line in curr_file:
		templine=line
		templine=templine.rstrip().replace(', ','')
		temp=templine.split(" ")
		curr_sa = temp[0] + ' ' + temp[3]
		if(curr_sa in demo_data):
			training_content += line[:-2] + '0\n'
		elif (temp[0] in demo_states): # action not found but state found
			training_content += line[:-2] + '10\n'
		elif temp[0] not in demo_states: #state not found
			training_content += line[:-2] + '0\n'
	curr_file.close()
	curr_file = open(training,"w")
	curr_file.write(training_content)
	curr_file.close()

####### BP
def boxpushing():
	to_write = "Problem, Budget, Technique, V1, Std Deviation, V2, Std Deviation\n "
	bp = test_boxpushing(sensitivity)
	testInstances = bp.getTestInstances()
	testprob = bp.gettestProb()
	trainprob = bp.getTrainProb()
	training = '--box=data/boxpushing/'+trainprob +".bp"
	gamma = '--gamma=0.95'

	global_slack = getSlack(training)
	for technique in techniques:
		print("********************************************",technique)
		epsilon = 0.1
		cf = '--cost='+technique
		if(technique == "RR" or technique == "HA-L" or technique == "HA-S"):
			call(['./testNSE.out',training,gamma,'--slack=0','--algorithm=OF','--numObj=2','--v=1',cf]) #Training samples
			trainFile = "data/boxpushing/"+ trainprob +"_Samples.txt"
			x_train,y_train,x_full = bp.ParseFiles(trainFile)
			for p in range(len(testprob)):
				if(p == 0):
					continue
				tprob = testInstances[p]
				slack = getSlack(tprob)
				testingFile, processedFile = bp.generate_proactive_samples(tprob,technique,p,slack)
				x_test,y_test,testInput = bp.ParseFiles(testingFile)

				for budgetVal in HA_budget:
					mse = LearnReward(budgetVal,processedFile,testInput,x_test, y_test, x_train, y_train) #Learn
					opline = bp.solve_learnedReward(tprob, testprob[p], processedFile, technique,slack) #Plan
					os.remove(processedFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')
					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])

					to_write += testprob[p] + ", "+ budgetVal +"," + technique + ", " + \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"

				os.remove(testingFile)
			os.remove(trainFile)

		elif(technique == "Demo-AM"):
			# trainig file = trajectories of opt policy. 
			for trials in HA_budget:
				# trFile, testingFile, processedFile = bp.generate_demo_samples(training,0,global_slack,trials)
				trFile = bp.generate_demo_samples(training,0,global_slack,trials)
				trFile = "data/boxpushing/"+ trainprob +"_Samples.txt"
				for p in range(len(testprob)):
					if(p == 0):
						continue
					tprob = testInstances[p]
					slack = getSlack(tprob)
					testingFile, processedFile = bp.generate_proactive_samples(tprob,"RR",p,slack)
					demoAcceptable(trFile,testingFile) #will modify training file as demo-acceptable
					x_train,y_train,x_full = bp.ParseFiles(trFile) 
					x_test,y_test,testInput = bp.ParseFiles(testingFile)
					mse = LearnReward(len(x_train),processedFile,testInput,x_test, y_test, x_train, y_train) #Learn
					opline = bp.solve_learnedReward(tprob, testprob[p], processedFile, technique,slack)#Plan
					
					os.remove(processedFile)
					os.remove(testingFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])

					to_write += testprob[p] + ", "+ trials +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(trFile)

		elif(technique == "correction"):
			# trainig file =  opt policy. 
			for trials in HA_budget:
				trFile, testingFile, processedFile = bp.generate_corrections(training,0,trials,global_slack)
				x_train,y_train,x_full = bp.ParseFiles(trFile)
				for p in range(len(testprob)):
					if(p == 0):
						continue
					tprob = testInstances[p]
					slack = getSlack(tprob)
					testingFile, processedFile = bp.generate_proactive_samples(tprob,technique,p,slack)
					x_test,y_test,testInput = bp.ParseFiles(testingFile)
					mse = LearnReward(len(x_train),processedFile,testInput,x_test, y_test, x_train, y_train) #Learn

					opline = bp.solve_learnedReward(tprob, testprob[p], processedFile, technique,slack)#Plan
					os.remove(processedFile)
					os.remove(testingFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
					
					to_write += testprob[p] + ", "+ budgetVal +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(trFile)

		elif (technique == "radical" or technique == "moderate" or technique == "conservative"):
			if(technique == "radical"):
				epsilon = 0.9
			elif(technique == "moderate"):
				epsilon = 0.5
				
				#Learn
			for episodes in RL_budget:
				print("episodes = ", episodes)
				trainFile, testingFile, processedFile = bp.generate_reactive_samples(training, epsilon, episodes, global_slack, 0)
				x_train,y_train,x_full = bp.ParseFiles(trainFile)
				for p in range(len(testprob)):
					if(p == 0):
						continue
					tprob = testInstances[p]
					slack = getSlack(tprob)
					testingFile, processedFile = bp.generate_proactive_samples(tprob,technique,p,slack)
					x_test,y_test,testInput = bp.ParseFiles(testingFile)
					mse = LearnReward(len(x_train),processedFile,testInput,x_test, y_test, x_train, y_train) 
					opline = bp.solve_learnedReward(tprob, testprob[p], processedFile, technique, slack)#Plan
					os.remove(processedFile)
					os.remove(testingFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
			
					to_write += testprob[p] + ", "+ episodes +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(trainFile)

		
		elif(technique == "deviation"):
			call(['./testNSE.out',training,gamma,'--slack=0','--algorithm=OF','--numObj=2','--v=1',"--cost=RR"]) #Training samples
			trainFile = "data/boxpushing/"+ trainprob +"_Samples.txt"
			x_train,y_train,x_full = bp.ParseFiles(trainFile)
			for p in range(len(testprob)):
				if(p == 0):
					continue

				tprob = testInstances[p]
				slack = getSlack(tprob)
				testingFile, processedFile = bp.generate_proactive_samples(tprob,technique,p,slack)
				x_test,y_test,testInput = bp.ParseFiles(testingFile)

				for budgetVal in HA_budget:
					mse = LearnReward(budgetVal,processedFile,testInput,x_test, y_test, x_train, y_train) #Learn
					rewardFile = '--testing='+processedFile
					opline=subprocess.check_output(['./testNSE.out',tprob,'--gamma=0.95','--algorithm=deviation','--numObj=3','--v=100',rewardFile,'--weights=1,0.8'])
					os.remove(processedFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
					to_write += testprob[p] + ", "+ budgetVal +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
					
				os.remove(testingFile)
			os.remove(trainFile)

		elif(technique == "Opt" or technique == "UB"):
			for p in range(len(testprob)):
				if(p == 0):
					continue
				tprob = testInstances[p]
				slack = getSlack(tprob)
				slackip = '--slack='+str(slack)

				if(technique == "Opt"):
					opline=subprocess.check_output(['./testNSE.out',tprob,gamma,slackip,'--algorithm=LLAO','--numObj=2','--v=100'])
				if(technique == "UB"):
					opline=subprocess.check_output(['./testNSE.out',tprob,gamma,'--slack=0','--algorithm=LLAO','--numObj=2','--v=100'])

				if not isinstance(opline, str):
					opline = opline.decode('utf-8')

				temp=opline.split("\n")
				v1,v2,stdV1,stdV2 = AvgReward(temp[1])
					
				to_write += testprob[p] + ", 0,"+ technique + ","+ \
				v1+","+stdV1+","+v2+","+stdV2 +"\n"

		elif technique == "IF":
			slack = 0
			for p in range(len(testprob)):
				if(p == 0):
					continue
				tprob = testInstances[p]

				opline=subprocess.check_output(['./testNSE.out',tprob,gamma,'--algorithm=LLAO','--numObj=1','--v=100','--impactFree'])

				if not isinstance(opline, str):
					opline = opline.decode('utf-8')

				temp=opline.split("\n")
				v1,v2,stdV1,stdV2 = AvgReward(temp[1])
	
				to_write += testprob[p] + ", 0,"+ technique + ","+ \
				v1+","+stdV1+","+v2+","+stdV2 +"\n"

	curr_file = open(sensitivity+"Boxpushing-Logs.csv","w")
	curr_file.write(to_write)
	curr_file.close()


#### nav
def navigation():
	to_write = "Problem, Budget, Technique, V1, Std Deviation, V2, Std Deviation\n "
	nav = test_navigation(sensitivity)
	testInstances = nav.getTestInstances()
	testprob = nav.gettestProb()
	trainprob = nav.getTrainProb()
	training = '--nav=data/navigation/'+trainprob +".nav"
	gamma = '--gamma=0.95'

	global_slack = getSlack(training)
	for technique in techniques:
		print("********************************************",technique)
		epsilon = 0.1
		mse = 0
		cf = '--cost='+technique
		if(technique == "RR" or technique == "HA-L" or technique == "HA-S"):
			call(['./testNSE.out',training,gamma,'--slack=0','--algorithm=OF','--numObj=2','--v=1',cf]) #Training samples
			trainFile = "data/navigation/"+ trainprob +"_Samples.txt"
			x_train,y_train,x_full = nav.ParseFiles(trainFile)
			for p in range(len(testprob)):
				if(p == 0):
					continue
				tprob = testInstances[p]
				slack = getSlack(tprob)
				testingFile, processedFile = nav.generate_proactive_samples(tprob,technique,p,slack)
				x_test,y_test,testInput = nav.ParseFiles(testingFile)

				for budgetVal in HA_budget:
					mse = LearnReward(budgetVal,processedFile,testInput,x_test, y_test, x_train, y_train) #Learn
					opline = nav.solve_learnedReward(tprob, testprob[p], processedFile, technique,slack) #Plan
					os.remove(processedFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
			
					to_write += testprob[p] + ", "+ budgetVal +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
					
				os.remove(testingFile)
			os.remove(trainFile)

		elif(technique == "Demo-AM"):
			# trainig file = trajectories of opt policy. 
			for trials in HA_budget:
				trFile, testingFile, processedFile = nav.generate_demo_samples(training,0,global_slack,trials)
				trFile = "data/navigation/"+ trainprob +"_Samples.txt"
				for p in range(len(testprob)):
					if(p == 0):
						continue
					tprob = testInstances[p]
					slack = getSlack(tprob)
					testingFile, processedFile = nav.generate_proactive_samples(tprob,"RR",p,slack)
					demoAcceptable(trFile,testingFile) #will modify training file as demo-acceptable
					x_train,y_train,x_full = nav.ParseFiles(trFile) 
					x_test,y_test,testInput = nav.ParseFiles(testingFile)
					mse = LearnReward(len(x_train),processedFile,testInput,x_test, y_test, x_train, y_train) #Learn
					opline = nav.solve_learnedReward(tprob, testprob[p], processedFile, technique,slack)#Plan
					
					os.remove(processedFile)
					os.remove(testingFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
					
					to_write += testprob[p] + ", "+ trials +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(trFile)

		elif(technique == "correction"):
			# trainig file =  opt policy. 
			for trials in HA_budget:
				trFile, testingFile, processedFile = nav.generate_corrections(training,0,trials,global_slack)
				x_train,y_train,x_full = nav.ParseFiles(trFile)
				for p in range(len(testprob)):
					if(p == 0):
						continue
					tprob = testInstances[p]
					slack = getSlack(tprob)
					testingFile, processedFile = nav.generate_proactive_samples(tprob,technique,p,slack)
					x_test,y_test,testInput = nav.ParseFiles(testingFile)
					mse = LearnReward(len(x_train),processedFile,testInput,x_test, y_test, x_train, y_train) #Learn

					# mse = LearnReward(100,processedFile,testInput,testInput, y_test, x_full, y_train) #Learn
					opline = nav.solve_learnedReward(tprob, testprob[p], processedFile, technique,slack)#Plan
					os.remove(processedFile)
					os.remove(testingFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
			
					to_write += testprob[p] + ", "+ trials +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(trFile)

		elif (technique == "radical" or technique == "moderate" or technique == "conservative"):
			if(technique == "radical"):
				epsilon = 0.9
			elif(technique == "moderate"):
				epsilon = 0.5
				
				#Learn
			for episodes in RL_budget:
				print("episodes = ", episodes)
				trainFile, testingFile, processedFile = nav.generate_reactive_samples(training, epsilon, episodes, global_slack, 0)
				x_train,y_train,x_full = nav.ParseFiles(trainFile)
				for p in range(len(testprob)):
					if(p == 0):
						continue
					tprob = testInstances[p]
					slack = getSlack(tprob)
					testingFile, processedFile = nav.generate_proactive_samples(tprob,technique,p,slack)
					x_test,y_test,testInput = nav.ParseFiles(testingFile)
					mse = LearnReward(len(x_train),processedFile,testInput,x_test, y_test, x_train, y_train) 
					opline = nav.solve_learnedReward(tprob, testprob[p], processedFile, technique, slack)#Plan
					os.remove(processedFile)
					os.remove(testingFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
			
					to_write += testprob[p] + ", "+ episodes +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(trainFile)

		elif(technique == "deviation"):
			# call(['./testNSE.out',training,gamma,'--slack=0','--algorithm=OF','--numObj=2','--v=1',cf]) #Training samples
			call(['./testNSE.out',training,gamma,'--slack=0','--algorithm=OF','--numObj=2','--v=1','--cost=RR']) #Training samples
			trainFile = "data/navigation/"+ trainprob +"_Samples.txt"
			x_train,y_train,x_full = nav.ParseFiles(trainFile)
			for p in range(len(testprob)):
				if(p == 0):
					continue
				
				tprob = testInstances[p]
				slack = getSlack(tprob)
				# slackip = '--slack='+str(slack)
				testingFile, processedFile = nav.generate_proactive_samples(tprob,technique,p,slack)
				x_test,y_test,testInput = nav.ParseFiles(testingFile)

				for budgetVal in HA_budget:
					mse = LearnReward(budgetVal,processedFile,testInput,x_test, y_test, x_train, y_train) #Learn
					rewardFile= '--testing='+processedFile
					opline=subprocess.check_output(['./testNSE.out',tprob,gamma,'--algorithm=deviation','--numObj=2','--v=100',rewardFile,'--weights=1,0.8'])
					os.remove(processedFile)

					if not isinstance(opline, str):
						opline = opline.decode('utf-8')

					temp=opline.split("\n")
					v1,v2,stdV1,stdV2 = AvgReward(temp[1])
					to_write += testprob[p] + ", "+ budgetVal +"," + technique + ","+ \
					v1+","+stdV1+","+v2+","+stdV2 +"\n"
				os.remove(testingFile)
			os.remove(trainFile)

		elif(technique == "Opt" or technique == "UB"):
			for p in range(len(testprob)):
				if(p == 0):
					continue
				tprob = testInstances[p]
				slack = getSlack(tprob)
				slackip = '--slack='+str(slack)

				if(technique == "Opt"):
					opline=subprocess.check_output(['./testNSE.out',tprob,gamma,slackip,'--algorithm=LLAO','--numObj=2','--v=100'])
				if(technique == "UB"):
					opline=subprocess.check_output(['./testNSE.out',tprob,gamma,'--slack=0','--algorithm=LLAO','--numObj=2','--v=100'])
				
				if not isinstance(opline, str):
					opline = opline.decode('utf-8')

				temp=opline.split("\n")
				v1,v2,stdV1,stdV2 = AvgReward(temp[1])	
				to_write += testprob[p] + ", 0,"+ technique + ","+ \
				v1+","+stdV1+","+v2+","+stdV2 +"\n"

		elif technique == "IF":
			slack = 0
			for p in range(len(testprob)):
				if(p == 0):
					continue
				tprob = testInstances[p]
				bias = 0

				opline=subprocess.check_output(['./testNSE.out',tprob,gamma,'--algorithm=LAO','--numObj=1','--v=100','--impactFree'])

				if not isinstance(opline, str):
					opline = opline.decode('utf-8')

				temp=opline.split("\n")
				v1,v2,stdV1,stdV2 = AvgReward(temp[1])
	
				to_write += testprob[p] + ", 0,"+ technique + ","+ \
				v1+","+stdV1+","+v2+","+stdV2 +"\n"

	curr_file = open(sensitivity+"Navigation-Logs.csv","w")
	curr_file.write(to_write)
	curr_file.close()



#remvoes all .pyc files generated
def clean():
	if os.path.exists('Regression.pyc'):
		os.remove('Regression.pyc')

	if os.path.exists('test_boxpushing.pyc'):
		os.remove('test_boxpushing.pyc')

	if os.path.exists('test_navigation.pyc'):
		os.remove('test_navigation.pyc')


def main():
	if(domain_name == "boxpushing"):
		boxpushing()

	if(domain_name == "navigation"):
		navigation()

	clean()



if __name__ == '__main__':
    main()