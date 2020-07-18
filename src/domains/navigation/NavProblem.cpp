/** Author: Sandhya Saisubramanian, UMass Amherst **/

#include <vector>
#include <cassert>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string.h>
#include <cctype>

#include "../../../include/domains/navigation/NavProblem.h"
#include "../../../include/domains/navigation/NavState.h"
#include "../../../include/domains/navigation/NavAction.h"

#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>



using namespace boost;
namespace mlmobj
{
NavProblem::NavProblem(const char* filename, int ipsize, const char* costFunction, bool useLearnedvalue, std::string reward_file)
{
   size_ = ipsize;
   costFunctionType_ = costFunction;
   weights_ = std::vector<double> (size_, 0.0);
   weights_[0] = 1.0;
   std::ifstream myfile (filename);
   goals_ = new PairDoubleMap();
   bool boxpresent = false;
    // Once the file is finished parsing, these will have correct values
    width_ = 0, height_ = 0;
    if (myfile.is_open()) {
        std::string line;
        while ( std::getline (myfile, line) ) {
            for (width_ = 0; width_ < line.size(); width_++) {
                if (line.at(width_) == 'x' || line.at(width_) == 'X') {
                    walls.insert(std::pair<int, int>(width_, height_));
                } else if (line.at(width_) == '@') {
                    water.insert(std::pair<int, int>(width_, height_));
                } else if (line.at(width_) == 'H'){
                    humanLocs.insert(std::pair<int, int>(width_, height_));
                } else if (line.at(width_) == 'G') {
                    goals_->insert(
                        std::make_pair(
                            std::pair<int,int> (width_, height_), 0.0));
                 } else if (line.at(width_) == 'S') {
                    x0_ = width_;
                    y0_ = height_;
                } else {
                    assert(line.at(width_) == '.');
                }
            }
            height_++;
        }
        myfile.close();
    } else {
        std::cerr << "Invalid file " << filename << std::endl;
        exit(-1);
    }
    s0 = new NavState(x0_, y0_, 0, false, false, this);
    absorbing = new NavState(-1, -1,-1, false, false, this);
    this->addState(s0);
    this->addState(absorbing);
     addAllActions();

    if(useLearnedvalue)
        LoadLearnedReward(reward_file);
}


void NavProblem::addAllActions()
{
    mlcore::Action* left_slow = new NavAction(AVNavigation::LEFT_SLOW);
    mlcore::Action* left_fast = new NavAction(AVNavigation::LEFT_FAST);
    mlcore::Action* right_slow = new NavAction(AVNavigation::RIGHT_SLOW);
    mlcore::Action* right_fast = new NavAction(AVNavigation::RIGHT_FAST);
    mlcore::Action* up_slow = new NavAction(AVNavigation::UP_SLOW);
    mlcore::Action* up_fast = new NavAction(AVNavigation::UP_FAST);
    mlcore::Action* down_slow = new NavAction(AVNavigation::DOWN_SLOW);
    mlcore::Action* down_fast = new NavAction(AVNavigation::DOWN_FAST);
    actions_.push_front(left_slow);
    actions_.push_front(left_fast);
    actions_.push_front(right_slow);
    actions_.push_front(right_fast);
    actions_.push_front(up_slow);
    actions_.push_front(up_fast);
    actions_.push_front(down_slow);
    actions_.push_front(down_fast);
}


void NavProblem::LoadLearnedReward(std::string reward_file)
{
    std::ifstream learnedReward_file;
    learnedReward_file.open(reward_file);
    int index = 0;
    std::string line;
    std::vector<std::string> vec;
    while (getline(learnedReward_file, line)) {
        std::istringstream iss(line);
         vec.clear();
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        std::string key = vec.at(0) + vec.at(1)+ vec.at(2)+vec.at(3)+vec.at(4)+ vec.at(5); //features of s,a without (x,y) location in s.
        double reward = stod(vec.at(6));
        learned_reward_.insert({key,fabs(reward)});
    }
    learnedReward_file.close();
}
//original costs
double NavProblem::cost(mlcore::State* s, mlcore::Action* a, int index) const
{
//    assert(index < size_);
    NavProblem* np = const_cast<NavProblem*> (this);
    NavState* ns = static_cast<NavState *>(s);
    NavAction* na = (NavAction *) a;

    int idAction = a->hashValue();
     if(index == 0 || index > 1) //return reduced model cost for o1
        return np->cost_reduced(s,a,0);

    if (s == absorbing || goal(s,0))
        return 0.0;

     //learned reward
    if(use_learnedReward_){
        std::string key = std::to_string(ns->x()) +" " +std::to_string(ns->y()) + " " + std::to_string(ns->curr_speed()) +" "
                         + std::to_string(ns->human()) + " "+ std::to_string(ns->water()) + " "+
                          std::to_string(idAction);
        auto it = learned_reward_.find(key);
        if (it != learned_reward_.end()){
            return learned_reward_.at(key);
        }
        else{
                return 0;
            }
    }

     //human acceptance-based  cost function
   // HC-L: lenient human classification HC-S: strict human classification
    if(strcmp(costFunctionType_, "RR") == 0 ){
        return np->HumanFeedback(s,a,1);
     }
    if(strcmp(costFunctionType_, "HA-L") == 0 || strcmp(costFunctionType_, "HA-S") == 0){
        if(np->HumanClassification(s,a,1))
            return 10.0;
        return 0;
    }

    //orginal costs of NSE.
    if(np->isHigh_NSE(s,a))
        return 10.0;
    if(np->isLow_NSE(s,a))
        return 5.0;

   return 0;
}
double NavProblem::cost(mlcore::State* s, mlcore::Action* a) const
{
    if(!deviation_score_){
        double lcCost = 0.0;
        for (int i = 0; i < size_; i++) {
            lcCost += weights_.at(i) * cost(s, a, i);
        }
        return lcCost;
    }

    //Deviation score is true.
    NavProblem* np = const_cast<NavProblem*> (this);
    NavState* ns = static_cast<NavState *>(s);
    int idAction = a->hashValue();
    double r = cost_reduced(s,a,0);
    double deviation = 0;

    if(use_learnedReward_){
        std::string key = std::to_string(ns->x()) +" " +std::to_string(ns->y()) + " " + std::to_string(ns->curr_speed()) +" "
                         + std::to_string(ns->human()) + " "+ std::to_string(ns->water()) + " "+
                          std::to_string(idAction);
        auto it = learned_reward_.find(key);
        if (it != learned_reward_.end()){
           if (learned_reward_.at(key) > 0)
                deviation = 1;
        }
    }
    else {
       if(np->isLow_NSE(s,a) || np->isHigh_NSE(s,a))
            deviation = 1;
        }

    // returns reward for optimizing a scalarized function based on deviation score.
    return weights_.at(0) * r + weights_.at(1) * deviation;
}

double NavProblem::cost_reduced(mlcore::State* s, mlcore::Action* a, int i) const
{
    assert(i < size_);

    if (s == absorbing || goal(s,0))
        return 0.0;
     NavState* ns = (NavState *) s;
     NavAction* na = (NavAction*) a;

    if(walls.count(std::pair<int, int> (ns->x(), ns->y())) != 0)
        return 5.0;

    if(na->dir() == AVNavigation::LEFT_SLOW || na->dir() == AVNavigation::RIGHT_SLOW ||
        na->dir() == AVNavigation::UP_SLOW || na->dir() == AVNavigation::DOWN_SLOW)
            return 2;

   return 1.0; // standard cost
}

bool NavProblem::isLow_NSE(mlcore::State* s, mlcore::Action* a)const{
    if (s == absorbing || goal(s,0))
        return false;

    NavState* ns = (NavState *) s;
    NavAction* na = (NavAction *) a;

   if(ns->water()){
    if(na->dir() == AVNavigation::LEFT_FAST || na->dir() == AVNavigation::RIGHT_FAST ||
        na->dir() == AVNavigation::UP_FAST || na->dir() == AVNavigation::DOWN_FAST)
        return true;
   }
    return false;
}

bool NavProblem::isHigh_NSE(mlcore::State* s, mlcore::Action* a)const{
    if (s == absorbing || goal(s,0))
        return false;

    NavState* ns = (NavState *) s;
    NavAction* na = (NavAction *) a;

    if(ns->human()){
    if(na->dir() == AVNavigation::LEFT_FAST || na->dir() == AVNavigation::RIGHT_FAST ||
        na->dir() == AVNavigation::UP_FAST || na->dir() == AVNavigation::DOWN_FAST)
        return true;
    }
    return false;
}

double NavProblem::HumanFeedback(mlcore::State* s, mlcore::Action* a, int index)const{

    assert(index < size_);

    if (s == absorbing || goal(s,0))
        return 0.0;

    NavState* ns = (NavState *) s;
    NavAction* na = (NavAction *) a;

    NavProblem* np = const_cast<NavProblem*> (this);
    if(np->isHigh_NSE(s,a))
        return 10.0;
    if(np->isLow_NSE(s,a))
        return 5.0;

    return 0;
}

bool NavProblem::HumanClassification(mlcore::State* s, mlcore::Action* a, int index)const{

    assert(index < size_);

    if (s == absorbing || goal(s,0))
        return false;

    NavState* ns = (NavState *) s;
    NavAction* na = (NavAction *) a;
    NavProblem* np = const_cast<NavProblem*> (this);

    if(np->isHigh_NSE(s,a))
        return true;
    // reaches here for strict human acceptance
    if(strcmp(costFunctionType_, "HA-S") == 0){
            if(np->isLow_NSE(s,a))
                return true;
    }

    return false;
}
bool NavProblem::goal(mlcore::State* s, int index) const
{
    NavState* bps = (NavState *) s;
    std::pair<int,int> pos(bps->x(),bps->y());
    if(goals_[index].find(pos) != goals_[index].end() )
        return true;
    return false;
}
bool NavProblem::applicable(mlcore::State* s, mlcore::Action* a) const
{
    NavState* ns = (NavState*) s;
    NavAction* na = (NavAction*) a;
    NavProblem* np = const_cast<NavProblem*> (this);
    if (s == absorbing)
        return true;
    /** This only allows actions in each state that do not result in any NSE **/
    if(impactFreeTrans_){
        if(np->isHigh_NSE(s,a) || np->isLow_NSE(s,a)){
//       if(na->dir() == AVNavigation::LEFT_FAST || na->dir() == AVNavigation::RIGHT_FAST ||
//        na->dir() == AVNavigation::UP_FAST || na->dir() == AVNavigation::DOWN_FAST){
            return false;
        }
    }

    return true;
}

bool NavProblem::applicable(mlcore::State* s, mlcore::Action* a, int index) const
{
    if(index == 0)
        return applicable(s,a);

    NavState* bps = (NavState*) s;
    std::list<mlcore::Action*> filteredActions = bps->filteredActions();
    if(filteredActions.size() == 0){
        std::cout << "filtered actions empty for " << s << std::endl;
        exit(EXIT_FAILURE) ;
    }
    for(auto it : filteredActions){
        if(it == a)
            return applicable(s,a);
    }
    return false;
}


std::list<mlcore::Successor>
NavProblem::transition(mlcore::State* s, mlcore::Action* a, int index)
{
    assert(applicable(s, a,index));
    NavState* state = (NavState *) s;
    NavAction* action = (NavAction *) a;
    NavProblem* np = const_cast<NavProblem*> (this);
    std::list<mlcore::Successor> successors;

    if (s == absorbing || goal(s,index)) {
        successors.push_front(mlcore::Successor(absorbing, 1.0));
        return successors;
    }

    int idAction = action->hashValue();
    std::vector<mlcore::SuccessorsList>* allSuccessors = state->allSuccessors();

    if (!allSuccessors->at(idAction).empty()) {
        return allSuccessors->at(idAction);
    }

    double successprob = 0.9;
    double failureprob = 0.1;
    bool curr_human = (humanLocs.count(std::pair<int, int> (state->x(), state->y())) != 0);
    bool curr_water = (water.count(std::pair<int, int> (state->x(), state->y())) != 0);

// don't worry about water or human features, add succ function will handle it.
   if(action->dir() == AVNavigation::LEFT_FAST || action->dir() == AVNavigation::LEFT_SLOW){

        int newspeed = (action->dir() == AVNavigation::LEFT_FAST )? 2 : 1;

        addSuccessor(state, successors, state->x() - 1, state->y(), newspeed, curr_human, curr_water,successprob);
        addSuccessor(state, successors, state->x(), state->y() + 1, newspeed, curr_human, curr_water,failureprob); //slides down
   }
   else if(action->dir() == AVNavigation::RIGHT_FAST || action->dir() == AVNavigation::RIGHT_SLOW){
        int newspeed = (action->dir() == AVNavigation::RIGHT_FAST )? 2 : 1;

        addSuccessor(state, successors, state->x() + 1, state->y(), newspeed, curr_human, curr_water,successprob);
        addSuccessor(state, successors, state->x(), state->y() + 1, newspeed, curr_human, curr_water,failureprob); //slides down
   }
   else if(action->dir() == AVNavigation::UP_FAST || action->dir() == AVNavigation::UP_SLOW){
        int newspeed = (action->dir() == AVNavigation::UP_FAST )? 2 : 1;

        addSuccessor(state, successors, state->x(), state->y() - 1, newspeed, curr_human, curr_water,successprob);
        addSuccessor(state, successors, state->x() + 1, state->y(), newspeed, curr_human, curr_water,failureprob); // slides right
   }
   else if(action->dir() == AVNavigation::DOWN_FAST || action->dir() == AVNavigation::DOWN_SLOW){
        int newspeed = (action->dir() == AVNavigation::DOWN_FAST )? 2 : 1;

        addSuccessor(state, successors, state->x(), state->y() + 1, newspeed, curr_human, curr_water,successprob);
        addSuccessor(state, successors, state->x() + 1, state->y(), newspeed, curr_human, curr_water,failureprob); // slides right
   }
    return successors;
}
void NavProblem::addSuccessor(NavState* state, std::list<mlcore::Successor>& successors,
                                    int newx, int newy, int newspeed,
                                    bool newhuman, bool newwater, double prob)
{
    bool isWall = (walls.count(std::pair<int, int> (newx, newy)) != 0);
    bool ishuman = (humanLocs.count(std::pair<int, int> (newx, newy)) != 0);
    bool iswater = (water.count(std::pair<int, int> (newx, newy)) != 0);


    if ((newx < 0 || newx >= width_) || (newy < 0 || newy >= height_) || isWall)
         successors.push_front(mlcore::Successor(state, prob));
    else{
     NavState *next = new NavState(newx, newy, newspeed, ishuman, iswater, this);
     successors.push_front(mlcore::Successor(this->addState(next), prob));
     }
}

}
