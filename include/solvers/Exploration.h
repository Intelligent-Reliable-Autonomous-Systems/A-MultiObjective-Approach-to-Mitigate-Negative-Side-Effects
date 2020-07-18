#ifndef EXPLORATION_H
#define EXPLORATION_H

#include <utility>
#include <unordered_map>
#include <vector>

#include "../MObjProblem.h"
#include "../MObjState.h"


  class Exploration
  {
    private:
        double epsilon_;
    		double gamma_;
    		mlmobj::MOProblem* problem_;
    		int max_steps = 20000;
        double final_score_ = 0;
        double exploration_anneal = 1000;
        std::string samples_file = "";
        bool generateSamples_ = true;
        std::string learner_writer="";

    public:
         Exploration(mlmobj::MOProblem* problem, double epsilon, std::string samples_file="");
		     ~Exploration() {};

    		double epsilon() { return epsilon_; }
    		void epsilon(double eps) {epsilon_ = eps; }

        bool generateSamples() {return generateSamples_;}
        void generateSamples(bool gs) { generateSamples_ = gs;}

        /** Returns the greedy action for the state **/
        virtual mlcore::Action* getGreedyAction(mlmobj::MOProblem* problem, mlmobj::MOState* s, int index);

        virtual mlmobj::MOState* getRandomSuccessor(mlmobj::MOProblem* problem, mlmobj::MOState* s, mlcore::Action* a);

		    virtual void gatherFeedback(mlmobj::MOProblem* problem, mlmobj::MOState* s, int index);

        virtual void writeSamples();

  };
#endif
