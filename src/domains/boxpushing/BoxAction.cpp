#include "../../../include/domains/boxpushing/BoxAction.h"
#include "../../../include/domains/boxpushing/BoxProblem.h"


std::ostream& BoxAction::print(std::ostream& os) const
{
    os << "action ";
    if (dir_ == box::UP)
        os << "up";
    if (dir_ == box::DOWN)
        os << "down";
    if (dir_ == box::LEFT)
        os << "left";
    if (dir_ == box::RIGHT)
        os << "right";
    if (dir_ == box::PICK)
        os << "pickup";
    return os;
}
