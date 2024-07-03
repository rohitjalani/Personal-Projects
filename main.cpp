#include <bits/stdc++.h>
#include "Model/RubiksCube1dArray.cpp"
#include "Solver/BFSSolver.h"
#include "Solver/DFSSolver.h"

using namespace std;

int main() 
{
   RubiksCube1dArray object1dArray;

   if (object1dArray.isSolved()) cout << "SOLVED\n\n";
   else cout << "NOT SOLVED\n\n";

   object1dArray.u();
   object1dArray.print();

   object1dArray.l();
   object1dArray.print();

   object1dArray.f();
   object1dArray.print();

   object1dArray.r();
   object1dArray.print();

   object1dArray.b();
   object1dArray.print();

   object1dArray.d();
   object1dArray.print();

   if (object1dArray.isSolved()) cout << "SOLVED\n\n";
   else cout << "NOT SOLVED\n\n";

   object1dArray.dPrime();
   object1dArray.print();

   object1dArray.bPrime();
   object1dArray.print();

   object1dArray.rPrime();
   object1dArray.print();

   object1dArray.fPrime();
   object1dArray.print();

   object1dArray.lPrime();
   object1dArray.print();

   object1dArray.uPrime();
   object1dArray.print();

   if (object1dArray.isSolved()) cout << "SOLVED\n\n";
   else cout << "NOT SOLVED\n\n";


   // Create two Cubes ------------------------------------------------------------------------------------------

   RubiksCube1dArray cube1;
   RubiksCube1dArray cube2;

  // Equality and assignment of cubes --------------------------------------------------------------------------

   if(cube1 == cube2) cout << "Is equal\n";
   else cout << "Not Equal\n";

   cube1.randomShuffleCube(1);

   if(cube1 == cube2) cout << "Is equal\n";
   else cout << "Not Equal\n";

   cube2 = cube1;

   if(cube1 == cube2) cout << "Is equal\n";
   else cout << "Not Equal\n";


  // Unordered_map of Cubes  ------------------------------------------------------------------------------------

   unordered_map<RubiksCube1dArray, bool, Hash1d> mp1;

   mp1[cube1] = true;
   cube2.randomShuffleCube(8);
   if (mp1[cube1]) cout << "Cube1 is present\n";
   else cout << "Cube1 is not present\n";

   if (mp1[cube2]) cout << "Cube2 is present\n";
   else cout << "Cube2 is not present\n";

// BFS Solver -----------------------------------------------------------------------------------------------------
   cout<<" -------------- BFS Solver -------- "<<"\n";
   RubiksCube1dArray cubeBFS;
   cubeBFS.print();

   vector<RubiksCube::MOVE> shuffle_moves = cubeBFS.randomShuffleCube(10);
   for(auto move: shuffle_moves) cout << cubeBFS.getMove(move) << " ";
   cout << "\n";
   cubeBFS.print();

   BFSSolver<RubiksCube1dArray, Hash1d> bfsSolver(cubeBFS);
   vector<RubiksCube::MOVE> solve_moves = bfsSolver.solve();

   for(auto move: solve_moves) cout << cubeBFS.getMove(move) << " ";
   cout<<"BFS Solved"<<"\n";
   cout << "\n";
   bfsSolver.rubiksCube.print();


   cout<<" -------------- DFS Solver -------- "<<"\n";
   RubiksCube1dArray cubeDFS;
   cubeDFS.print();

   vector<RubiksCube::MOVE> shuffle_moves = cubeDFS.randomShuffleCube(10);
   for(auto move: shuffle_moves)
   {
      cout<<cubeDFS.getMove(move)<<" ";
   }
   cout<<"\n";
   cubeDFS.print();

   DFSSolver<RubiksCube1dArray, Hash1d> dfsSolver(cubeDFS);
   vector<RubiksCube::MOVE> solve_moves = dfsSolver.solve();

   for(auto move: solve_moves)
   {
      cout<<cubeDFS.getMove(move)<<" ";
   }
   cout<<"DFS Soolved"<<"\n";
   cout<<"\n";
   dfsSolver.rubiksCube.print();
   
   return 0;
}