#ifndef RUBIKS_CUBE_SOLVER_RUBIKSCUBE_H // Check whether previously macro has been imported or not to avoid error due to redundancy
#define RUBIKS_CUBE_SOLVER_RUBIKSCUBE_H

#include <bits/stdc++.h>
using namespace std;

class RubiksCube 
{

public:
// enum class: This is a keyword in C++ that lets you create a set of named values, all grouped together under a single type.
    enum class FACE
    {
        UP,
        LEFT,
        FRONT,
        RIGHT,
        BACK,
        DOWN
    };

    enum class COLOR
    {
        WHITE,
        GREEN,
        RED,
        BLUE,
        ORANGE,
        YELLOW
    };

/*
Basic Moves
L: Rotate the left face 90 degrees clockwise.
R: Rotate the right face 90 degrees clockwise.
U: Rotate the upper face 90 degrees clockwise.
D: Rotate the bottom (down) face 90 degrees clockwise.
F: Rotate the front face 90 degrees clockwise.
B: Rotate the back face 90 degrees clockwise.
Prime Moves
Prime moves (denoted with PRIME) are the counterclockwise rotations of the corresponding face:

LPRIME: Rotate the left face 90 degrees counterclockwise.
RPRIME: Rotate the right face 90 degrees counterclockwise.
UPRIME: Rotate the upper face 90 degrees counterclockwise.
DPRIME: Rotate the bottom (down) face 90 degrees counterclockwise.
FPRIME: Rotate the front face 90 degrees counterclockwise.
BPRIME: Rotate the back face 90 degrees counterclockwise.
Double Moves
Double moves (denoted with 2) rotate the corresponding face 180 degrees:

L2: Rotate the left face 180 degrees.
R2: Rotate the right face 180 degrees.
U2: Rotate the upper face 180 degrees.
D2: Rotate the bottom (down) face 180 degrees.
F2: Rotate the front face 180 degrees.
B2: Rotate the back face 180 degrees.
*/
    enum class MOVE
    {
        L, LPRIME, L2,
        R, RPRIME, R2,
        U, UPRIME, U2,
        D, DPRIME, D2,
        F, FPRIME, F2,
        B, BPRIME, B2
    };

    /*
     * Returns the color of the cell at (row, col) in face.
     * If Rubik's Cube face is pointing at you, then the row numbering starts from the
     * top to bottom, and column numbering starts from the left to right.
     * The rows and columns are 0-indexed.
     * @param Face, row, and column index
    */
    virtual COLOR getColor(FACE face, int row, int col) const = 0; // const = 0 means its a pure virtual function
    // Here COLOR was the return type, and getColor is a function

    /*
     * Returns the first letter of the given COLOR
     * Eg: For COLOR::GREEN, it returns 'G'
    */

//static keyword: Allows the function to be called using the class name itself without needing an instance. Without it, the function must be called on an instance of the class.
    static char getColorLetter(COLOR color);

    /*
     * Returns true if the Rubik Cube is solved, otherwise returns false.
    */
    virtual bool isSolved() const = 0;

   /*
    * returns the move in the string format
   */
    static string getMove(MOVE ind);

    void print() const;

   /*
    * Randomly shuffle the cube with 'times' moves and returns the moves performed.
   */
    vector<MOVE> randomShuffleCube(int times);

   /*
    * Perform moves on the Rubik Cube
   */
    RubiksCube &move(MOVE ind); //A way to declare member function(move)

  /*
    * Invert a move
  */
    RubiksCube &invert(MOVE ind);

  /*
     * Rotational Moves on the Rubik Cubes
     *
     * F, F’, F2,
     * U, U’, U2,
     * L, L’, L2,
     * D, D’, D2,
     * R, R’, R2,
     * B, B’, B2
  */

    virtual RubiksCube &f() = 0;
    virtual RubiksCube &fPrime() = 0;
    virtual RubiksCube &f2() = 0;
    virtual RubiksCube &u() = 0;
    virtual RubiksCube &uPrime() = 0;
    virtual RubiksCube &u2() = 0;
    virtual RubiksCube &l() = 0;
    virtual RubiksCube &lPrime() = 0;
    virtual RubiksCube &l2() = 0;
    virtual RubiksCube &d() = 0;
    virtual RubiksCube &dPrime() = 0;
    virtual RubiksCube &d2() = 0;
    virtual RubiksCube &r() = 0;
    virtual RubiksCube &rPrime() = 0;
    virtual RubiksCube &r2() = 0;
    virtual RubiksCube &b() = 0;
    virtual RubiksCube &bPrime() = 0;
    virtual RubiksCube &b2() = 0;

    string getCornerColorString(int ind) const;
    int getCornerIndex(int ind) const;
    int getCornerOrientation(int ind) const;
};

#endif