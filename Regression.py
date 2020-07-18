import sys
import random
import numpy as np
import statistics
from subprocess import call
# import subprocess
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterSampler
# from sklearn.tree import export_graphviz

class Regression:
	def __init__(self):
		self.params = {"min_samples_split": [2, 10, 20,50],
					   "max_depth": [None, 2, 5, 10],
						"max_features":['auto'],
					   "min_samples_leaf": [1, 5, 10, 50],
					   "max_leaf_nodes": [None, 5, 10, 20],
					   "n_estimators": [10, 50, 80,100],
					   "random_state":[None,0]}
		self.x_train=[]
		self.y_train=[]
		self.error_val=[]
		self.budget=[]
		self.reg = RandomForestRegressor()
		self.final_model = RandomForestRegressor()

	def train_folds(self,model,x_train_, y_train_, x_test_, y_test_): 
		model.fit(x_train_,y_train_)
		test_label = model.predict(x_test_)
		mse = mean_squared_error(test_label, y_test_)
		return mse


	def Predict(self,x_train,y_train,x_test,y_test,testInput, processedFile, budget):
		limit = min(int(budget),len(x_train))
		indices = random.sample(range(0,len(x_train)),int(limit))
		budget_xtrain = []
		budget_ytrain = []
		for i in indices:
			budget_xtrain.append(x_train[i])
			budget_ytrain.append(y_train[i])

		x=np.array(budget_xtrain)
		y=np.array(budget_ytrain)
		candidate_params = list(ParameterSampler(param_distributions=self.params, n_iter=10, random_state=None))
		model = clone(self.reg)
		num_splits = 3
		if(sum(budget_ytrain) < num_splits):
			num_splits =  int(sum(budget_ytrain))
		if(sum(budget_ytrain) <= 1):
			num_splits = 1
		best_params = None
		best_score = -1
		# Loop through all possible parameter configurations and find the best one
		for parameters in candidate_params:
			model.set_params(**parameters)
			if(num_splits > 1):
				cv =  StratifiedKFold(n_splits=num_splits,random_state= None,shuffle=True)
				cv_scores  = []
			
				for train,test in cv.split(x, y):
					x_train_, x_test_, y_train_, y_test_ =  x[train], x[test], y[train], y[test]
					score = self.train_folds(model,x_train_,y_train_,x_test_,y_test_)
					cv_scores.append(score)
				avg_score = float(sum(cv_scores))/len(cv_scores)
				if(avg_score > best_score):
					best_params = parameters
					best_score = avg_score
			else:
				best_params = parameters
				break


		self.final_model = clone(model)
		self.final_model.set_params(**best_params)
		self.final_model.fit(budget_xtrain,budget_ytrain)


		test_label = self.final_model.predict(x_test)
		mse = mean_squared_error(test_label,y_test)
		myfile = open(processedFile,"w")
		for i in range(len(testInput)):
			f = ''.join(str(testInput[i]))
			f = f.replace('[','').replace(']','')
			myfile.write(f+",")
			myfile.write(str(test_label[i])+"\n")
		myfile.close()
		return mse



