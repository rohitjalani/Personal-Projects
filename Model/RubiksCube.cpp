#include "RubiksCube.h"

/*
Given a color, return its first letter
*/

char RubiksCube::getColorLetter(COLOR color)
{
    if(color == COLOR::BLUE)  // if(color == RubiksCube::COLOR::BLUE) can also be used, but here only single header file is imported, so without RubiksCube:: also it will work
    {
        return 'B';
    }
    else if(color == COLOR::GREEN)
    {
        return 'G';
    }
    else if(color == COLOR::RED)
    {
        return 'R';
    }
    else if(color == COLOR::YELLOW)
    {
        return 'Y';
    }
    else if(color == COLOR::WHITE)
    {
        return 'W';
    }
    else
    {
        return 'O';
    }
}

/*
Describe a move using an index
*/

string RubiksCube::getMove(MOVE ind)
{
    if(ind == MOVE::L)
    {
        return "L";
    }
    else if(ind == MOVE::LPRIME)
    {
        return "L'";
    }
    else if(ind == MOVE::L2)
    {
        return "L2";
    }
    else if(ind == MOVE::R)
    {
        return "R";
    }
    else if(ind == MOVE::RPRIME)
    {
        return "R'";
    }
    else if(ind == MOVE::R2)
    {
        return "R2";
    }
    else if(ind == MOVE::U)
    {
        return "U";
    }
    else if(ind == MOVE::UPRIME)
    {
        return "U'";
    }
    else if(ind == MOVE::U2)
    {
        return "U2";
    }
    else if(ind == MOVE::D)
    {
        return "D";
    }
    else if(ind == MOVE::DPRIME)
    {
        return "D'";
    }
    else if(ind == MOVE::D2)
    {
        return "D2";
    }
    else if(ind == MOVE::B)
    {
        return "B";
    }
    else if(ind == MOVE::BPRIME)
    {
        return "B'";
    }
    else if(ind == MOVE::B2)
    {
        return "B2";
    }
    else if(ind == MOVE::F)
    {
        return "F";
    }
    else if(ind == MOVE::FPRIME)
    {
        return "F'";
    }
    else
    {
        return "F2";
    }
}

/*
Perform a move operation on using a Move index
*/

RubiksCube &RubiksCube::move(MOVE ind) //this calls move function from RubiksCube.h, which overriden by u() function in RubiksCube1dArray.cpp
{
    if(ind == MOVE::L)
    {
        return this->l();
    }
    else if(ind == MOVE::LPRIME)
    {
        return this->lPrime();
    }
    else if(ind == MOVE::L2)
    {
        return this->l2();
    }
    else if(ind == MOVE::R)
    {
        return this->r();
    }
    else if(ind == MOVE::RPRIME)
    {
        return this->rPrime();
    }
    else if(ind == MOVE::R2)
    {
        return this->r2();
    }
    else if(ind == MOVE::U)
    {
        return this->u();
    }
    else if(ind == MOVE::UPRIME)
    {
        return this->uPrime();
    }
    else if(ind == MOVE::U2)
    {
        return this->u2();
    }
    else if(ind == MOVE::D)
    {
        return this->d();
    }
    else if(ind == MOVE::DPRIME)
    {
        return this->dPrime();
    }
    else if(ind == MOVE::D2)
    {
        return this->d2();
    }
    else if(ind == MOVE::F)
    {
        return this->f();
    }
    else if(ind == MOVE::FPRIME)
    {
        return this->fPrime();
    }
    else if(ind == MOVE::F2)
    {
        return this->f2();
    }
    else if(ind == MOVE::B)
    {
        return this->b();
    }
    else if(ind == MOVE::BPRIME)
    {
        return this->bPrime();
    }
    else
    {
        return this->b2();
    }
}

/*
Invert a move
*/

RubiksCube &RubiksCube::invert(MOVE ind)
{
    if(ind == MOVE::L)
    {
        return this->lPrime();
    }
    else if(ind == MOVE::LPRIME)
    {
        return this->l();
    }
    else if(ind == MOVE::L2)
    {
        return this->l2();
    }
    else if(ind == MOVE::R)
    {
        return this->rPrime();
    }
    else if(ind == MOVE::RPRIME)
    {
        return this->r();
    }
    else if(ind == MOVE::R2)
    {
        return this->r2();
    }
    else if(ind == MOVE::U)
    {
        return this->uPrime();
    }
    else if(ind == MOVE::UPRIME)
    {
        return this->u();
    }
    else if(ind == MOVE::U2)
    {
        return this->u2();
    }
    else if(ind == MOVE::D)
    {
        return this->dPrime();
    }
    else if(ind == MOVE::DPRIME)
    {
        return this->d();
    }
    else if(ind == MOVE::D2)
    {
        return this->d2();
    }
    else if(ind == MOVE::F)
    {
        return this->fPrime();
    }
    else if(ind == MOVE::FPRIME)
    {
        return this->f();
    }
    else if(ind == MOVE::F2)
    {
        return this->f2();
    }
    else if(ind == MOVE::B)
    {
        return this->bPrime();
    }
    else if(ind == MOVE::BPRIME)
    {
        return this->b();
    }
    else
    {
        return this->b2();
    }
}


/*
Print Rubik's Cube
*/

void RubiksCube::print() const
{
    cout<<"Rubiks Cube\n\n";

    for(int row=0;row<=2;row++)
    {
        for(int i=0;i<7;i++)
        {
            cout<<" ";
        }
        for(int col=0;col<=2;col++)
        {
            cout<<getColorLetter(getColor(FACE::UP, row, col))<<" ";
        }
        cout<<"\n";
    }

    cout<<"\n";

    for(int row=0;row<=2;row++)
    {
        for(int col=0;col<=2;col++)
        {
            cout<<getColorLetter(getColor(FACE::LEFT, row, col))<<" ";
        }
        cout<<" ";

        for(int col=0;col<=2;col++)
        {
            cout<<getColorLetter(getColor(FACE::FRONT, row, col))<<" ";
        }
        cout<<" ";

        for(int col=0;col<=2;col++)
        {
            cout<<getColorLetter(getColor(FACE::RIGHT, row, col))<<" ";
        }
        cout<<" ";

        for(int col=0;col<=2;col++)
        {
            cout<<getColorLetter(getColor(FACE::BACK, row, col))<<" ";
        }
        cout<<"\n";
    }

    cout<<"\n";

    for (int row = 0; row <= 2; row++) 
    {
        for (unsigned i = 0; i < 7; i++) cout << " ";
        for (int col = 0; col <= 2; col++) 
        {
            cout << getColorLetter(getColor(FACE::DOWN, row, col)) << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

vector<RubiksCube::MOVE> RubiksCube::randomShuffleCube(int times)
{
    vector<MOVE> moves_performed;
    srand(time(0));
    for(int i=0;i<times;i++)
    {
        int selectMove = (rand() % 18);
        moves_performed.push_back(static_cast<MOVE>(selectMove));
        this->move(static_cast<MOVE>(selectMove));
        // moves_performed.push_back(static_cast<MOVE>(selectMove));
        // this->move(static_cast<MOVE>(selectMove));
    }
    return moves_performed;
}

string RubiksCube::getCornerColorString(int ind) const
{
    string str = "";

    // UFR
    if(ind == 0)
    {
        str += getColorLetter(getColor(FACE::UP,2,2));
        str += getColorLetter(getColor(FACE::FRONT,0,2));
        str += getColorLetter(getColor(FACE::RIGHT,0,0));
    }

    // UFL
    if(ind == 1)
    {
        str += getColorLetter(getColor(FACE::UP,2,0));
        str += getColorLetter(getColor(FACE::FRONT,0,0));
        str += getColorLetter(getColor(FACE::LEFT,0,2));
    }

    // UBL
    if(ind == 2)
    {
        str += getColorLetter(getColor(FACE::UP,0,0));
        str += getColorLetter(getColor(FACE::FRONT,0,2));
        str += getColorLetter(getColor(FACE::LEFT,0,0));
    }

    // UBR
    if(ind == 3)
    {
        str += getColorLetter(getColor(FACE::UP,0,2));
        str += getColorLetter(getColor(FACE::FRONT,0,0));
        str += getColorLetter(getColor(FACE::RIGHT,0,2));
    }

    // DFR
    if(ind == 4)
    {
        str += getColorLetter(getColor(FACE::DOWN,0,2));
        str += getColorLetter(getColor(FACE::FRONT,2,2));
        str += getColorLetter(getColor(FACE::RIGHT,2,0));
    }

    // DFL
    if(ind == 5)
    {
        str += getColorLetter(getColor(FACE::DOWN,0,0));
        str += getColorLetter(getColor(FACE::FRONT,2,0));
        str += getColorLetter(getColor(FACE::LEFT,2,2));
    }

    // DBR
    if(ind == 6)
    {
        str += getColorLetter(getColor(FACE::DOWN,2,2));
        str += getColorLetter(getColor(FACE::BACK,2,0));
        str += getColorLetter(getColor(FACE::RIGHT,2,2));
    }

    // DBL
    if(ind == 4)
    {
        str += getColorLetter(getColor(FACE::DOWN,2,0));
        str += getColorLetter(getColor(FACE::FRONT,2,2));
        str += getColorLetter(getColor(FACE::LEFT,2,0));
    }

    return str;
}


/*
Background
A Rubik's Cube has 8 corners, and each corner has 3 stickers. Each sticker can be one of 6 colors. The combination of colors on each corner can be used to uniquely identify that corner.

Bitwise Encoding
In this function, we use a 3-bit binary number to represent each corner. Each bit in the 3-bit number corresponds to the presence of a specific color pair:

First bit (least significant bit): Represents the presence of green (G) among blue (B) and green (G).
Second bit: Represents the presence of orange (O) among red (R) and orange (O).
Third bit (most significant bit): Represents the presence of yellow (Y) among white (W) and yellow (Y).
By setting these bits accordingly, we can create a unique binary number (and thus a unique integer index) for each possible corner configuration.
*/
int RubiksCube::getCornerIndex(int ind) const
{
    string corner = getCornerColorString(ind);

    int ret = 0;
    for(auto c: corner)
    {
        if(c!='W' && c!='Y')
        {
            continue;
        }
        if(c == 'Y')
        {
            ret |= (1<<2);
        }
    }

    for(auto c: corner)
    {
        if(c!='R' && c!='O')
        {
            continue;
        }
        if(c == 'O')
        {
            ret |= (1<<1);
        }
    }

    for(auto c: corner)
    {
        if(c!='B' && c!='G')
        {
            continue;
        }
        if(c == 'G')
        {
            ret |= (1<<0);
        }
    }

    return ret;
}


int RubiksCube::getCornerOrientation(int ind) const
{
    string corner = getCornerColorString(ind);
    string actual_str = "";

    for(auto c: corner)
    {
        if(c!='W' && c!='Y')
        {
            continue;
        }
        actual_str.push_back(c);
    }

    if(corner[1] == actual_str[0])
    {
        return 1;
    }
    else if(corner[2] == actual_str[0])
    {
        return 2;
    }
    else
    {
        return 0;
    }
}