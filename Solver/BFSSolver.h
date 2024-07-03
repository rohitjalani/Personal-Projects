#include <bits/stdc++.h>
#include "../Model/RubiksCube.h"

#ifndef RUBIKS_CUBE_SOLVER_BFSSOLVER_H
#define RUBIKS_CUBE_SOLVER_BFSSOLVER_H

/*
T: Represents the Rubik's Cube representation (3D, 1D, Bitboard, etc.).
H: Represents the hash function corresponding to the Rubik's Cube representation.
*/
template<typename T, typename H>
class BFSSolver
{
private:

/*
moves: Stores the sequence of moves to solve the Rubik's Cube.
visited: An unordered map to keep track of visited cube states to avoid cycles.
move_done: An unordered map that stores the move performed to reach each state, used to reconstruct the solution path.
*/
    vector<RubiksCube::MOVE> moves;
    unordered_map<T, bool, H> visited;
    unordered_map<T, RubiksCube::MOVE, H> move_done;

    T bfs()
    {
        queue<T> q;
        q.push(rubiksCube);
        visited[rubiksCube] = true;

        while(!q.empty())
        {
            T node = q.front();
            q.pop();
            if(node.isSolved())
            {
                return node;
            }

            for(int i=0;i<18;i++)
            {
                auto curr_move = RubiksCube::MOVE(i);
                node.move(curr_move);
                if(!visited[node])
                {
                    visited[node] = true;
                    move_done[node] = curr_move;
                    q.push(node);
                }
                node.invert(curr_move);
            }
        }
        return rubiksCube;
    }

public:
    T rubiksCube;
    BFSSolver(T _rubiksCube)
    {
        rubiksCube = _rubiksCube;
    }

    vector<RubiksCube::MOVE> solve()
    {
        T solved_cube = bfs();
        assert(solved_cube.isSolved());
        T curr_cube = solved_cube;
        while(!(curr_cube == rubiksCube))
        {
            RubiksCube::MOVE curr_move = move_done[curr_cube];
            moves.push_back(curr_move);
            curr_cube.invert(curr_move);
        }
        rubiksCube = solved_cube;
        reverse(moves.begin(),moves.end());
        return moves;
    }
};

#endif 