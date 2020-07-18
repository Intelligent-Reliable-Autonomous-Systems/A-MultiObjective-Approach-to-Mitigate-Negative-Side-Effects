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

#include "../../../include/domains/boxpushing/BoxProblem.h"
#include "../../../include/domains/boxpushing/BoxState.h"
#include "../../../include/domains/boxpushing/BoxAction.h"

#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>



using namespace boost;
namespace mlmobj
{
BoxProblem::BoxProblem(const char* filename, int ipsize, const char* costFunction, bool useLearnedvalue, std::string reward_file)
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
                if (line.at(width_) == 'x') {
                    walls.insert(std::pair<int, int>(width_, height_));
                } else if (line.at(width_) == '@') {
                    holes.insert(std::pair<int, int>(width_, height_));
                } else if (line.at(width_) == 'F') {
                    fragiles.insert(std::pair<int, int>(width_, height_));
                } else if (line.at(width_) == 'G') {
                    goals_->insert(
                        std::make_pair(
                            std::pair<int,int> (width_, height_), 0.0));
                 } else if (line.at(width_) == 'B') {
                    bx_0_ = width_;
                    by_0_ =  height_;
                    boxpresent = true;
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


    s0 = new BoxState(x0_, y0_,false, false, false, this);
    absorbing = new BoxState(-1, -1,true, false, false, this);
    this->addState(s0);
    this->addState(absorbing);
    addAllActions();
    if(useLearnedvalue)
        LoadLearnedReward(reward_file);
}
void BoxProblem::addAllActions()
{
    mlcore::Action* up = new BoxAction(box::UP);
    mlcore::Action* down = new BoxAction(box::DOWN);
    mlcore::Action* left = new BoxAction(box::LEFT);
    mlcore::Action* right = new BoxAction(box::RIGHT);
    mlcore::Action* pick = new BoxAction(box::PICK);

    actions_.push_front(up);
    actions_.push_front(down);
    actions_.push_front(left);
    actions_.push_front(right);
    actions_.push_front(pick);

}

void BoxProblem::LoadLearnedReward(std::string reward_file)
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
        std::string key = vec.at(0) + vec.at(1) + vec.at(2)+vec.at(3)+vec.at(4)+ vec.at(5);
        double reward = stod(vec.at(6));
        learned_reward_.insert({key,fabs(reward)});
    }
    learnedReward_file.close();
}
//original costs
double BoxProblem::cost(mlcore::State* s, mlcore::Action* a, int index) const
{
    BoxProblem* bp = const_cast<BoxProblem*> (this);
    BoxState* bps = static_cast<BoxState *>(s);
    int idAction = a->hashValue();
     if(index == 0 || index > 1) //return reduced model cost for o1
        return bp->cost_reduced(s,a,0);

    if (s == absorbing || goal(s,0))
        return 0.0;

     //learned reward
    if(use_learnedReward_){
        std::string key = std::to_string(bps->x()) + " " + std::to_string(bps->y()) + " " +
                          std::to_string(bps->loaded()) + " "+ std::to_string(bps->hole()) + " "+
                          std::to_string(bps->fragile()) + " " +std::to_string(idAction);
        auto it = learned_reward_.find(key);
        if (it != learned_reward_.end()){
            return learned_reward_.at(key);
        }
        else
            return 0;
    }
   //human acceptance-based  cost function
    if(strcmp(costFunctionType_, "RR") == 0){
        double to_ret =  0.0;
        std::vector<double>* approved = bps->allApproval();
        if (approved->at(idAction) != -1)
            to_ret = (bp->HumanFeedback(s,a,1));
        else{
            to_ret = (bp->HumanFeedback(s,a,1));
            approved->at(idAction) = (bp->HumanFeedback(s,a,1));
            bp->IncrementQueryCount();
        }
        return to_ret;
    }

   // HC-L: lenient human classification HC-S: strict human classification
    if(strcmp(costFunctionType_, "HA-L") == 0 || strcmp(costFunctionType_, "HA-S") == 0){
        double to_ret = (bp->HumanClassification(s,a,1))? 10.0 : 0.0;
        return to_ret;
    }

    //orginal costs of NSE.
    if(bp->isHigh_NSE(s,a))
        return 10;
    if(bp->isLow_NSE(s,a))
        return 5;

   return 0;
}
double BoxProblem::cost(mlcore::State* s, mlcore::Action* a) const
{
   if(!deviation_score_){
        double lcCost = 0.0;
        for (int i = 0; i < size_; i++) {
            lcCost += weights_.at(i) * cost(s, a, i);
        }
        return lcCost;
    }

    // deviation score is true.
    BoxProblem* bp = const_cast<BoxProblem*> (this);
    BoxState* bps = static_cast<BoxState *>(s);
    int idAction = a->hashValue();
    double deviation = 0;

    if(use_learnedReward_){
        std::string key = std::to_string(bps->x()) + " " + std::to_string(bps->y()) + " " +
                          std::to_string(bps->loaded()) + " "+ std::to_string(bps->hole()) + " "+
                          std::to_string(bps->fragile()) + " " +std::to_string(idAction);
        auto it = learned_reward_.find(key);
        if (it != learned_reward_.end()){
           if (learned_reward_.at(key) > 0)
                deviation = 1;
        }
    }
    else {
        if(bp->isLow_NSE(s,a) || bp->isHigh_NSE(s,a))
            deviation = 1;
    }

    // returns reward for optimizing a scalarized function based on deviation score.
    return weights_.at(0) * cost_reduced(s,a,0) + weights_.at(1) * deviation;
}

double BoxProblem::cost_reduced(mlcore::State* s, mlcore::Action* a, int i) const
{
    assert(i < size_);

    if (s == absorbing || goal(s,0))
        return 0.0;
   return 1.0; // standard cost
}

bool BoxProblem::isLow_NSE(mlcore::State* s, mlcore::Action* a)const{
    if (s == absorbing || goal(s,0))
        return false;

    BoxState* bps = (BoxState *) s;
    if(bps->fragile() && bps->loaded())
        return true;
    return false;
}

bool BoxProblem::isHigh_NSE(mlcore::State* s, mlcore::Action* a)const {
    if (s == absorbing || goal(s,0))
        return false;

    BoxState* bps = (BoxState *) s;
    if(bps->hole() && bps->loaded())
        return true;
    return false;
}

double BoxProblem::HumanFeedback(mlcore::State* s, mlcore::Action* a, int index)const{

    assert(index < size_);

    if (s == absorbing || goal(s,0))
        return 0.0;

    BoxState* bps = (BoxState *) s;
    BoxAction* bpa = (BoxAction *) a;

    if(strcmp(costFunctionType_, "RR") == 0){
        if(bpa->dir() != box::PICK){
            if(bps->fragile() && bps->loaded())
                return 5.0;
            if(bps->hole()  && bps->loaded())
                return 10.0;
        }
    }
    if(bpa->dir() != box::PICK){
        if(bps->hole()  && bps->loaded())
            return 10.0;
        }
    return 0;
}

bool BoxProblem::HumanClassification(mlcore::State* s, mlcore::Action* a, int index)const{

    assert(index < size_);

    if (s == absorbing || goal(s,0))
        return false;

    BoxState* bps = (BoxState *) s;
    BoxAction* bpa = (BoxAction *) a;

   if(bps->hole() && bps->loaded() && bpa->dir() != box::PICK)
        return true;

    // reaches here for strict human acceptance
    if(strcmp(costFunctionType_, "HA-S") == 0){
        if(bpa->dir() != box::PICK){
            if(bps->fragile()  && bps->loaded())
                return true;
        }
    }
    return false;
}
bool BoxProblem::goal(mlcore::State* s, int index) const
{
    BoxState* bps = (BoxState *) s;
    std::pair<int,int> pos(bps->x(),bps->y());
    if(goals_[index].find(pos) != goals_[index].end() && bps->loaded())
        return true;
    return false;
}
bool BoxProblem::applicable(mlcore::State* s, mlcore::Action* a) const
{
    BoxState* bps = (BoxState*) s;
    BoxAction* bpa = (BoxAction*) a;
    BoxProblem* bp = const_cast<BoxProblem*> (this);
    if (s == absorbing)
        return true;

   if(bpa->dir() == box::PICK){
        if(!bps->loaded() && bps->x() == bx_0_ && bps->y() == by_0_ )
            return true;
        return false;
    }

     /** This only allows actions in each state that do not result in any NSE **/
    if(impactFreeTrans_){
        if(bp->isHigh_NSE(s,a) || bp->isLow_NSE(s,a)){
            return false;
        }
    }

    if((bps->x() + 1 >= width_ || (walls.count(std::pair<int, int> (bps->x()+1, bps->y())) != 0)) && bpa->dir() == box::RIGHT)
        return false;
    if((bps->x() - 1 < 0 || (walls.count(std::pair<int, int> (bps->x()-1, bps->y())) != 0)) && bpa->dir() == box::LEFT)
        return false;
    if((bps->y() - 1 < 0 || (walls.count(std::pair<int, int> (bps->x(), bps->y()-1)) != 0)) && bpa->dir() == box::UP)
        return false;
    return true;
}

bool BoxProblem::applicable(mlcore::State* s, mlcore::Action* a, int index) const
{
    if(index == 0)
        return applicable(s,a);

    BoxState* bps = (BoxState*) s;
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
BoxProblem::transition(mlcore::State* s, mlcore::Action* a, int index)
{
    std::list<mlcore::Successor> successors;
     if(impactFreeTrans_){
        BoxProblem* bp = const_cast<BoxProblem*> (this);
        if(bp->isHigh_NSE(s,a) || bp->isLow_NSE(s,a)){
            successors.push_front(mlcore::Successor(s, 1.0));
        return successors;
        }
    }

    assert(applicable(s, a));
    BoxState* state = (BoxState *) s;
    BoxAction* action = (BoxAction *) a;
    int idAction = action->hashValue();

    double prob_success = 0.95;
    double prob_slide = 0.05;

    std::vector<mlcore::SuccessorsList>* allSuccessors = state->allSuccessors();

    if (s == absorbing || goal(s,index)) {
        allSuccessors->at(idAction).push_back(mlcore::Successor(s, 1.0));
        return allSuccessors->at(idAction);
    }

    if(action->dir() == box::PICK){
        addSuccessor(state, successors, -1, -2, state->x(), state->y(), true, state->hole(), state->fragile(), 1.0);
        return successors;
    }
     if (!allSuccessors->at(idAction).empty()) {
        return allSuccessors->at(idAction);
    }

    if (action->dir() == box::DOWN) {
        addSuccessor(state, successors, height_ - 1, state->y(),
                     state->x(), state->y() + 1, state->loaded(), state->hole(), state->fragile(), prob_success);

        addSuccessor(state, successors, state->x(), 0,
                     state->x() - 1, state->y(), state->loaded(), state->hole(), state->fragile(),prob_slide);

        addSuccessor(state, successors, width_ - 1, state->x(),
                     state->x() + 1, state->y(), state->loaded(), state->hole(), state->fragile(), prob_slide);
    } else if (action->dir() == box::UP) {
        addSuccessor(state, successors, state->y(), 0,
                     state->x(), state->y() - 1, state->loaded(), state->hole(), state->fragile(), prob_success);

        addSuccessor(state, successors, state->x(), 0,
                     state->x() - 1, state->y(), state->loaded(), state->hole(),  state->fragile(), prob_slide);

        addSuccessor(state, successors, width_ - 1, state->x(),
                     state->x() + 1, state->y(), state->loaded(), state->hole(), state->fragile(), prob_slide);
    } else if (action->dir() == box::LEFT) {
        addSuccessor(state, successors, state->x(), 0,
                     state->x() - 1, state->y(), state->loaded(), state->hole(), state->fragile(), prob_success);

        addSuccessor(state, successors, state->y(), 0,
                     state->x(), state->y() - 1, state->loaded(), state->hole(), state->fragile(), prob_slide);

        addSuccessor(state, successors, height_ - 1, state->y(),
                     state->x(), state->y() + 1, state->loaded(), state->hole(), state->fragile(), prob_slide);
    } else if (action->dir() == box::RIGHT) {
        addSuccessor(state, successors, width_ - 1, state->x(),
                     state->x() + 1, state->y(), state->loaded(), state->hole(), state->fragile(),prob_success);

        addSuccessor(state, successors, state->y(), 0,
                     state->x(), state->y() - 1, state->loaded(), state->hole(), state->fragile(), prob_slide);

        addSuccessor(state, successors, height_ - 1, state->y(),
                     state->x(), state->y() + 1, state->loaded(), state->hole(), state->fragile(), prob_slide);
    }

    for(auto it =  successors.begin(); it != successors.end(); ++it)
        allSuccessors->at(idAction).push_back(*it);

    return successors;
}
void BoxProblem::addSuccessor(BoxState* state, std::list<mlcore::Successor>& successors,
                                    int val, int limit, int newx, int newy, bool newloaded, bool hole, bool fragile, double prob)
{
    bool isWall = (walls.count(std::pair<int, int> (newx, newy)) != 0);
    bool isHole = false;
    bool isFragile = false;
    if (holes.count(std::pair<int, int> (newx, newy)) != 0)
        isHole = true;
    if (fragiles.count(std::pair<int, int> (newx, newy)) != 0)
        isFragile = true;

    if (val > limit && !isWall) {
        BoxState *next = new BoxState(newx, newy, newloaded, isHole, isFragile, this);
        successors.push_front(mlcore::Successor(this->addState(next), prob));
    } else {
        successors.push_front(mlcore::Successor(state, prob));
    }
}

}
