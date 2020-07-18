#include <list>
#include <climits>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../../include/solvers/Exploration.h"
#include "../../include/MObjProblem.h"
#include "../../include/MObjState.h"

std::random_device rand_dev;

std::mt19937 kRNG(rand_dev());

std::uniform_real_distribution<> kUnif_0_1(0, 1);

using namespace std;

    Exploration::Exploration(mlmobj::MOProblem* problem, double epsilon, std::string sfile)
    {
        epsilon_ = epsilon;
        gamma_ = problem->gamma();
        samples_file = sfile;
        problem_ = problem;
    }

    mlcore::Action* Exploration::getGreedyAction(mlmobj::MOProblem* problem, mlmobj::MOState* s, int index) {
        double max_val = std::numeric_limits<double>::infinity();
        mlcore::Action* to_ret = nullptr;
        double rn = ((double)rand() / (RAND_MAX));
        std::vector<mlcore::Action*> valid_ac;
        valid_ac.clear();
        if (rn < epsilon_) {
            for(mlcore::Action* a: problem->actions()){
                if(problem->applicable(s,a, index)){
                    valid_ac.push_back(a);
                    }
                }
            int random_index = rand() % valid_ac.size();
            return valid_ac.at(random_index);
        }
        if(s->bestAction() != nullptr)
           return s->bestAction();
        else{
             for(mlcore::Action* a: problem->actions()){
                if(problem->applicable(s,a, index)){
                        return a;
                    }
                }

        }
        return to_ret;
    }

    mlmobj::MOState* Exploration::getRandomSuccessor(mlmobj::MOProblem* problem, mlmobj::MOState* s, mlcore::Action* a){

        if(problem->applicable(s,a)){
            double acc = 0.0;
            double rn = kUnif_0_1(kRNG);
            for (mlcore::Successor sccr : problem->transition(s, a)) {
                    acc += sccr.su_prob;
                    if(acc >= rn){
                        mlmobj::MOState* mos = static_cast<mlmobj::MOState*> (sccr.su_state);
                        return (mos);
                     }
                }
            }
        return s;
    }

/** Writes the samples (s,a,reward) to the samples_file**/
    void Exploration::writeSamples(){
            std::ofstream expfile;
            expfile.open(samples_file);
            expfile << learner_writer;
            expfile.close();
            std::cout << "Sample generation complete. Check " <<  samples_file << "." << std::endl;
    }

    void Exploration::gatherFeedback(mlmobj::MOProblem* problem, mlmobj::MOState* s, int index) {
        mlmobj::MOState* current_state = s;
    	int step = 0;

    	while (step < max_steps && !problem->goal(current_state,0)) {
    		step++;
    		mlcore::Action* current_action = getGreedyAction(problem, current_state, index);
            mlmobj::MOState* succ = getRandomSuccessor(problem, current_state, current_action);
    		double reward = problem->cost(current_state, current_action, index);
            if(generateSamples_){
                std::ostringstream curr_s, curr_a;
                curr_s << current_state;
                curr_a << current_action;
                learner_writer += curr_s.str() + " " + curr_a.str() + " " + std::to_string(current_action->hashValue())+ " " + std::to_string(reward) +"\n";
            }
        	current_state = succ;
    	}
    }
