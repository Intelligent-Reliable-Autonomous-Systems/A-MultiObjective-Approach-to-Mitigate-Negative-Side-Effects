#include <climits>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <string>

#include "../include/domains/boxpushing/BoxAction.h"
#include "../include/domains/boxpushing/BoxState.h"
#include "../include/domains/boxpushing/BoxProblem.h"
#include "../include/domains/navigation/NavProblem.h"
#include "../include/domains/navigation/NavAction.h"
#include "../include/domains/navigation/NavState.h"

#include "../include/solvers/Solver.h"
#include "../include/solvers/Exploration.h"
#include "../include/solvers/MOLAOStarSolver.h"
#include "../include/solvers/LAOStarSolver.h"
#include "../include/util/general.h"
#include "../include/util/flags.h"
#include "../include/util/graph.h"


#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
using namespace boost::accumulators;

using namespace mlsolvers;
using namespace mdplib_mobj_solvers;
using namespace mdplib;
using namespace mlcore;
using namespace mlmobj;
using namespace std;

int verbosity = 0;
MOProblem* problem = nullptr;
std::string testing_file = "";
std::string training_file = "";
std::string RL_training_file = "";
std::string processed_file = "";
std::string policy_file = "";
bool useLearnedValues = false;
int max_episodes=200;
Solver* solver = nullptr;
string problem_name  = "";
///////////////////////////////////////////////////////////////////////////////
//                              PROBLEM SETUP                                //
///////////////////////////////////////////////////////////////////////////////
void setupBoxPushing(){
    string boxFile = flag_value("box");
    testing_file = flag_value("box").substr(0, flag_value("box").find(".bp")) + "_Testing.txt";
    training_file = flag_value("box").substr(0, flag_value("box").find(".bp")) + "_Samples.txt";
    RL_training_file = flag_value("box").substr(0, flag_value("box").find(".bp")) + "_RLSamples.txt";
    processed_file = flag_value("box").substr(0, flag_value("box").find(".bp")) + "_Processed.csv";
    policy_file = flag_value("box").substr(0, flag_value("box").find(".bp")) + "_Policy.txt";

    int numObj = stoi(flag_value("numObj"));
    std::vector<double> slack(numObj, 0);

    if(flag_is_registered_with_value("slack")){
        std::string line = flag_value("slack");
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        int slack_length = vec.size();
        int limit = std::min(numObj, slack_length);
        for(int i = 0; i < limit; i++){
            slack.at(i) = stod(vec.at(i));
        }
    }
    double gamma = 0.99;
    if(flag_is_registered_with_value("gamma"))
        gamma = stod(flag_value("gamma"));

    string costfunc = "optimal";
    if(flag_is_registered_with_value("cost"))
        costfunc = flag_value("cost");

    if(flag_is_registered_with_value("testing")){
        useLearnedValues = true;
        processed_file = flag_value("testing");
    }
    problem = new BoxProblem(boxFile.c_str(), numObj, costfunc.c_str(), useLearnedValues, processed_file);
    problem->slack(slack);
    problem->gamma(gamma);
    problem->ProblemName("boxPushing");
    problem->StateDimensions(3); // used for RL algorithm function approximation
    if(useLearnedValues)
         problem->use_learnedReward(useLearnedValues);
    if(flag_is_registered("impactFree")){
        BoxProblem* bp = static_cast<BoxProblem*> (problem);
        bp->Set_impactFreeTrans(true);
    }
}
void setupNavigation(){
    string navFile = flag_value("nav");
    testing_file = flag_value("nav").substr(0, flag_value("nav").find(".nav")) + "_Testing.txt";
    training_file = flag_value("nav").substr(0, flag_value("nav").find(".nav")) + "_Samples.txt";
    RL_training_file = flag_value("nav").substr(0, flag_value("nav").find(".nav")) + "_RLSamples.txt";
    processed_file = flag_value("nav").substr(0, flag_value("nav").find(".nav")) + "_Processed.csv";
    policy_file = flag_value("nav").substr(0, flag_value("nav").find(".nav")) + "_Policy.txt";

    int numObj = stoi(flag_value("numObj"));
    std::vector<double> slack(numObj, 0);

    if(flag_is_registered_with_value("slack")){
        std::string line = flag_value("slack");
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        int slack_length = vec.size();
        int limit = std::min(numObj, slack_length);
        for(int i = 0; i < limit; i++){
            slack.at(i) = stod(vec.at(i));
        }
    }
    double gamma = 0.99;
    if(flag_is_registered_with_value("gamma"))
        gamma = stod(flag_value("gamma"));

    string costfunc = "optimal";
    if(flag_is_registered_with_value("cost"))
        costfunc = flag_value("cost");


    if(flag_is_registered_with_value("testing")){
        useLearnedValues = true;
        processed_file = flag_value("testing");
    }
    problem = new NavProblem(navFile.c_str(), numObj, costfunc.c_str(), useLearnedValues, processed_file);
    problem->slack(slack);
    problem->gamma(gamma);
    problem->ProblemName("navigation");
    problem->StateDimensions(3); // used for RL algorithm function approximation
    if(useLearnedValues)
         problem->use_learnedReward(useLearnedValues);
    if(flag_is_registered("impactFree")){
        NavProblem* np = static_cast<NavProblem*> (problem);
        np->Set_impactFreeTrans(true);
    }
}

void setupProblem()
{
    if (verbosity > 100)
        cout << "Setting up problem" << endl;
    if (flag_is_registered_with_value("box")) {
        problem_name = "boxpushing";
        setupBoxPushing();
    } else if(flag_is_registered_with_value("nav")){
        problem_name = "navigation";
        setupNavigation();
    }

    else {
        cerr << "Invalid problem." << endl;
        exit(-1);
    }
}
///////////////////////////////////////////////////////////////////////////////
//                               Exploration                                     //
///////////////////////////////////////////////////////////////////////////////
void setupExploration(string alg, Exploration*& learner){
    double epsilon = 0.1;
    if(flag_is_registered_with_value("epsilon"))
        epsilon = stod(flag_value("epsilon"));
    learner = new Exploration(problem,epsilon,RL_training_file);
    if(useLearnedValues)
        learner->generateSamples(false);
}


///////////////////////////////////////////////////////////////////////////////
//                              SOLVER SETUP                                 //
///////////////////////////////////////////////////////////////////////////////

void setupSolver(string alg, Solver*& solver){
    if (alg == "LLAO"){
        solver = new MOLAOStarSolver(problem);
    }
    
    else if(alg == "deviation"){
        problem->deviationScore("True");
        std::vector<double> weights;
        if(flag_is_registered_with_value("weights")){
            std::string line = flag_value("weights");
            std::vector<std::string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(","));
            int wt_length = vec.size();
            for(int i = 0; i < wt_length; i++)
                weights.push_back(stod(vec.at(i)));
        }
        else{
            weights.push_back(1); // weights for action reward wrt. o_1
            // corresponds to beta value in "Penalizing side effects using stepwise relative reachability"
            weights.push_back(0.1);
        }
        problem->weights(weights);
        solver = new MOLAOStarSolver(problem,false,1.0e-6,1000000,true);
    }
    else if(alg == "LAO"){
        double tol = 1.0e-3;
        solver = new LAOStarSolver(problem,tol, 100000000);
    }
    else{
        std::cout << " Algorithm not found" << std::endl;
        exit(1);
    }
}
/** Generates samples with true labels and creates a file for learning (with all s,a and without labels) **/
void generateSamples(int trials, Solver*& temp_solver){
    std::cout << " generating trajectories for querying " << std::endl;
    ofstream output_file;
    output_file.open(training_file);
    int trial_count = 0;
    mlcore::State* s = problem->initialState();
    int traj_length = 0;
    while(trial_count < trials){
        double prob = 0.0;
        mlcore::Action* a = s->bestAction();
        output_file << s << " " << (a)->hashValue() << std::endl;
        mlcore::State* succ = mlsolvers::randomSuccessor(problem,s,a, &prob);
        traj_length ++;
        s = succ;
        if(problem->goal(succ) || traj_length > 500){
            trial_count++;
            s = problem->initialState();
        }
    }
     output_file.close();
}
/* Generates all (s,a).. A part of this will be used for training, remaining will be used for testing*/
void generateSamples(){
    ofstream output_file;
     output_file.open(training_file);
     for(mlcore::State* s : problem->states()){
        mlmobj::MOState* mos = static_cast<mlmobj::MOState*>(s);
        std::list<mlcore::Action*> filteredActions = mos->filteredActions();
        for(mlcore::Action* a : filteredActions){
        // for(mlcore::Action* a : problem->actions()){
            if(problem->applicable(s,a)){
                 output_file << s << " " << a << " " << a->hashValue() << " " <<
                                problem->cost(s,a,1) << std::endl;
            }
        }
     }
     output_file.close();
}
void generateSamples(std::string filename){
    ofstream output_file;
     output_file.open(filename);
     for(mlcore::State* s : problem->states()){
        for(mlcore::Action* a : problem->actions()){
            if(problem->applicable(s,a)){
                 output_file << s << " " << a << " " << a->hashValue() << " " <<
                                problem->cost(s,a,1) << std::endl;
            }
        }
     }
     output_file.close();
}

void printPolicy(){
    ofstream output_file;
    output_file.open(policy_file);
    for(mlcore::State* s : problem->states()){
        if(s->bestAction() != nullptr){
            mlcore::Action* a = s->bestAction();
            output_file << s << " " << a << " " << a->hashValue() << " " << problem->cost(s,a,1) << std::endl;
        }
    }
    output_file.close();
}
///////////////////////////////////////////////////////////////////////////////
//                            SIMULATE                                       //
///////////////////////////////////////////////////////////////////////////////
//Simulates the computed policy
void simulate(int max_trials){
    int low_nse = 0, high_nse = 0; // tracks the number of low and high-impact nse (s,a) encountered.
    problem->use_learnedReward(false); 

    if(problem->getProblemName() == "boxPushing"){
        BoxProblem* bp = static_cast<BoxProblem*>(problem);
        bp->ResetCostFunction(); //resets using HA values.
    }
    else if(problem->getProblemName() == "navigation"){
        NavProblem* np = static_cast<NavProblem*>(problem);
        np->ResetCostFunction(); //resets using HA values.
    }
    vector <double> o1_cost, o2_cost;
    vector <double> lowNse, highNse;
    double TrialCost_1 = 0, TrialCost_2 = 0;

    for (int i = 0; i < max_trials; i++) {
        TrialCost_1 = 0, TrialCost_2 = 0;
        low_nse = 0, high_nse = 0;
        mlcore::State* tmp = problem->initialState();
        while (!problem->goal(tmp)) {
                Action* a = greedyAction(problem, tmp);
                 if(problem->getProblemName() == "boxPushing"){
                        BoxProblem* bp = static_cast<BoxProblem*>(problem);
                        if(bp->isLow_NSE(tmp,a))
                            low_nse++;
                        if(bp->isHigh_NSE(tmp,a))
                            high_nse++;
                    }
                else if(problem->getProblemName() == "navigation"){
                      NavProblem* cp = static_cast<NavProblem*>(problem);
                        if(cp->isLow_NSE(tmp,a)){
                            low_nse++;
                          }
                        if(cp->isHigh_NSE(tmp,a))
                            high_nse++;
                    }
                
                TrialCost_1 += problem->cost(tmp,a,0) * problem->gamma();
                TrialCost_2 +=  problem->cost(tmp,a,1) * problem->gamma();
                double prob = 0.0;
                State* aux = randomSuccessor(problem, tmp, a, &prob);
                tmp = aux;
                if(TrialCost_1 >= 500)
                    break;

            }
            o1_cost.push_back(TrialCost_1);
            o2_cost.push_back(TrialCost_2);
            lowNse.push_back(low_nse);
            highNse.push_back(high_nse);
    }
    accumulator_set<double, stats<tag::variance> > acc_o1_cost;
    for (auto &u : o1_cost){acc_o1_cost(u);}
    accumulator_set<double, stats<tag::variance> > acc_o2_cost;
    for (auto &u : o2_cost){acc_o2_cost(u);}

    accumulator_set<double, stats<tag::variance> > acc_lowNse;
    for (auto &u : lowNse){acc_lowNse(u);}
    accumulator_set<double, stats<tag::variance> > acc_highNse;
    for (auto &u : highNse){acc_highNse(u);}

    cout << "Avg. Exec cost " << mean(acc_o1_cost) << "  " << mean(acc_o2_cost);
    cout << "  Std. Dev. " << sqrt(variance(acc_o1_cost)) << "  " << sqrt(variance(acc_o2_cost)) << endl;

    cout << "Avg. NSE " << mean(acc_lowNse) << "  " << mean(acc_highNse);
    cout << "  Std. Dev. " << sqrt(variance(acc_lowNse)) << "  " << sqrt(variance(acc_highNse)) << endl;
}

void Exploration_feedback(int max_trials,Exploration*& learner){
    mlmobj::MOState* mos = static_cast<mlmobj::MOState*> (problem->initialState());
    for (int trial = 0; trial < max_trials; trial++) {
        for (int episodes = 0; episodes < max_episodes; episodes++) {
            learner->gatherFeedback(problem, mos,1);
        }
     }
     if(!useLearnedValues)
        learner->writeSamples();
}
///////////////////////////////////////////////////////////////////////////////
//                            MAIN                                           //
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* args[]){
    register_flags(argc, args);
    setupProblem();
    problem->generateAll();
    assert(flag_is_registered_with_value("algorithm"));
    string algorithm = flag_value("algorithm");
    if (flag_is_registered_with_value("v"))
        verbosity = stoi(flag_value("v"));
    if(flag_is_registered_with_value("episodes"))
        max_episodes = stoi(flag_value("episodes"));

    if(algorithm == "Exploration"){
        Solver* temp_solver = new MOLAOStarSolver(problem,true);  /* solve for level 0 optimally */
        temp_solver->solve(problem->initialState());
        delete temp_solver;
        Exploration* learner = nullptr;
        setupExploration(algorithm, learner);
        Exploration_feedback(1,learner);
        delete learner;
    }
    else if (algorithm == "OF") // Oracle Feedback
    {
        if(useLearnedValues) // plan with learned values
        {
            Solver* solver = new MOLAOStarSolver(problem);
            double start_time = clock();
            solver->solve(problem->initialState());
            double end_time = clock();
            std::cout << " total time taken (s) = " << (end_time-start_time)/ CLOCKS_PER_SEC << std::endl;
            int max_trials = 100;
            simulate(max_trials);
        }
        else{
            Solver* temp_solver = new MOLAOStarSolver(problem,true);  /* solve for level 0 optimally */
            temp_solver->solve(problem->initialState());
            double trials = 10;
            if (flag_is_registered_with_value("samples"))
                trials = stoi(flag_value("samples"));

            double budget = 100000000;
            if (flag_is_registered_with_value("budget"))
                budget = stoi(flag_value("budget"));

            generateSamples();
            delete temp_solver;
        }

    }else if(algorithm == "LLAO" && problem->size() == 1){
         Solver* temp_solver = new MOLAOStarSolver(problem,true);  /* solve for level 0 optimally */
         temp_solver->solve(problem->initialState());
         double return_val = ((MOState *) problem->initialState())->mobjCost()[0];
         std::cout << return_val << std::endl;
        if(verbosity >= 100)
            simulate(100);
    }

    else{
        setupSolver(algorithm, solver);
        solver->solve(problem->initialState());
        if(algorithm == "LAO"){
            std::cout << problem->initialState()->cost() << std::endl;
        }
        else{
            std::cout << " Estimated cost "
             << ((MOState *) problem->initialState())->mobjCost()[0] << " "
             << ((MOState *) problem->initialState())->mobjCost()[1] << endl;
         }
        if(algorithm == "LLAO" && flag_is_registered("demo")){
            int trials = 10;
           if (flag_is_registered_with_value("trials"))
                trials = stoi(flag_value("trials"));
             generateSamples(trials,solver);
             generateSamples(testing_file);
         }
         if(algorithm == "LAO" && flag_is_registered("demo")){
            int trials = 10;
           if (flag_is_registered_with_value("trials"))
                trials = stoi(flag_value("trials"));
             generateSamples(trials,solver);
             generateSamples(testing_file);
         }

        if(algorithm == "LLAO" && flag_is_registered("corrections") ){
           int trials = 10;
           if (flag_is_registered_with_value("trials"))
                trials = stoi(flag_value("trials"));
             generateSamples(trials,solver);
             generateSamples(testing_file);
        }

        if(verbosity > 200)
            printPolicy();

        if(verbosity >= 100){
            simulate(100);
            printPolicy();
        }
        if(verbosity >= 10)
            generateSamples(testing_file);
    }
    if(algorithm == "NLLAO" && !flag_is_registered("demo")){
        int lownse_count = 0, highnse_count = 0, sa_count = 0;
        for(mlcore::State* s: problem->states()){
            for(mlcore::Action* a: problem->actions()){
                if(problem->applicable(s,a)){
                    sa_count++;
                    mlmobj::MOState* mos = static_cast<mlmobj::MOState*> (s);
                    if(problem->isLow_NSE(s,a))
                        lownse_count++;
                    if(problem->isHigh_NSE(s,a))
                        highnse_count ++;
                }
            }
        }

        std::cout << "Low Nse fraction= " << (double(lownse_count)/double(sa_count))
                    << "  High Nse fraction= " << (double(highnse_count)/double(sa_count)) << std::endl;
    }

    delete solver;
    delete problem;
}
