#include <bits/stdc++.h>
#include "../Model/RubiksCube.h"

#ifndef RUBIKS_CUBE_SOLVER_DFSSOLVER_H
#define RUBIKS_CUBE_SOLVER_DFSSOLVER_H

template<typename Y, typename H>
class DFSSolver
{
private:
    vector<RubiksCube::MOVE> moves;
    int max_search_depth;

    bool dfs(int dep)
    {
        if(rubiksCube.isSolved())
        {
            return true;
        }
        if(dep>max_search_depth)
        {
            return false;
        }
        for(Int i=0;i<18;i++)
        {
            rubiksCube.move(RubiksCube::MOVE(i));
            moves.push_back(RubiksCube::MOVE(i));
            if(dfs(dep+1))
            {
                return true;
            }
            moves.pop_back();
            rubiksCube.invert(RubiksCube::MOVBE(i));
        }
        return false;
    }

public:
    T rubiksCube;

    //constructor
    DFSSolver(T _rubiksCube, int _max_search_depth=8)
    {
        rubiksCube = _rubiksCube;
        max_search_depth = _max_search_depth;
    }

    vector<RubiksCube::MOVE> solve()
    {
        dfs(1);
        return moves;
    }
};

#endif