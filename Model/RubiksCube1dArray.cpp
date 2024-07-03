/*
The functions/parameters that are obtained directly from the cube, are defined here
Like index, color, etc. Because all other information can be obtained from backend(RubiksCube.cpp), once we get these info
*/

#include "RubiksCube.h"
class RubiksCube1dArray : public RubiksCube
{
private:
/*
computes the index of a specific cell in the 1D array based on the face index (ind), row, and column.
*/
    static inline int getIndex(int ind, int row, int col)
    {
        return (ind*9)+(row*3)+col;
    }

    void rotateFace(int ind) //anticlockwise
    {
        char temp_arr[9] = {};
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                temp_arr[i*3+j] = cube[getIndex(ind, i, j)];
            }
        }

        for(int i=0;i<3;i++)
        {
            cube[getIndex(ind, 0, i)] = temp_arr[getIndex(0, 2-i, 0)]; //top row becomes left column

        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(ind, i, 2)] = temp_arr[getIndex(0, 0, i)]; //right column becomes top row

        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(ind, 2, 2-i)] = temp_arr[getIndex(0, i, 2)]; //bottom row becomes right column

        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(ind, 02-i, 0)] = temp_arr[getIndex(0, 2, 2-i)]; //left column becomes bottom row

        }
    }

public:
    char cube[54]{}; //stores all faces

    // Initializes cube to a solves state with each face having uniform color
    RubiksCube1dArray()
    {
        for(int i=0;i<6;i++)
        {
            for(int j=0;j<3;j++)
            {
                for(int k=0;k<3;k++)
                {
                    cube[i*9 + j*3 + k] = getColorLetter(COLOR(i));
                }
            }
        }
    }

/*
The getColor method returns the color of a given face, row, and column as a COLOR enum based on the character stored in the cube array.
*/
    COLOR getColor(FACE face, int row, int col) const override
    {
        char color = cube[getIndex((int)face,row,col)];
        if(color == 'B')
        {
            return COLOR::BLUE;
        }
        else if(color == 'R')
        {
            return COLOR::RED;
        }
        else if(color == 'G')
        {
            return COLOR::GREEN;
        }
        else if(color == 'O')
        {
            return COLOR::ORANGE;
        }
        else if(color == 'Y')
        {
            return COLOR::YELLOW;
        }
        else
        {
            return COLOR::WHITE;
        }
    }

    bool isSolved() const override 
    {
        for(int i=0;i<6;i++)
        {
            for(int j=0;j<3;j++)
            {
                for(int k=0;k<3;k++)
                {
                    if(this->cube[getIndex(i, j, k)] == getColorLetter(COLOR(i))) continue;
                    return false;
                }
            }
        }
        return true;
    }

    RubiksCube &u() override
    {

 /*
Rotate the Upper Face:
The method starts by calling rotateFace(0), which rotates the upper face (face 0) of the Rubik's Cube 90 degrees clockwise.
Adjust Edge Pieces:
The next steps involve rotating the edge pieces of the adjacent faces (left, front, right, and back) that are adjacent to the upper face. 
The edge pieces of these faces need to be swapped in a clockwise manner.
*/

        this->rotateFace(0);

        char temp_arr[3] = {};
        for(int i=0;i<3;i++)
        {
            temp_arr[i] = cube[getIndex(4, 0, 2-i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(4, 0, 2-i)] = cube[getIndex(1, 0, 2-i)];  // 4 means back face
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(1, 0, 2-i)] = cube[getIndex(2, 0, 2-i)]; // 1 means left face
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(2, 0, 2-i)] = cube[getIndex(3, 0, 2-i)]; // 2 means right face
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(3, 0, 2-i)] = temp_arr[i]; // 3 means front face
        }

/*
In the context of the u method in your RubiksCube1dArray class, returning *this ensures that after performing the upper face rotation, the same Rubik's Cube object is returned, enabling further operations to be called on it.
i.e. updates everything, then return pointer pointing to same cube
*/
        return *this;
    }

/*
The uPrime() method represents a counterclockwise 90-degree rotation of the upper face. This is achieved by calling the u() method three times, since three 90-degree clockwise rotations are equivalent to one 90-degree counterclockwise rotation.
*/
    RubiksCube &uPrime() override 
    {
        this->u();
        this->u();
        this->u();

        return *this;
    }

    RubiksCube &u2() override
    {
        this->u();
        this->u();

        return *this;
    }


    RubiksCube &l() override 
    {
        this->rotateFace(1);

        char temp_arr[3] = {};
        for(int i=0;i<3;i++)
        {
            temp_arr[i] = cube[getIndex(0, i, 0)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(0, i, 0)] = cube[getIndex(4, 2-i, 2)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(4, 2-i, 2)] = cube[getIndex(5, i, 0)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(5, i, 0)] = cube[getIndex(2, i, 0)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(2, i, 0)] = temp_arr[i];
        }
        return *this;
    }

    RubiksCube &lPrime() override 
    {
        this->l();
        this->l();
        this->l();

        return *this;
    }

    RubiksCube &l2() override
    {
        this->l();
        this->l();

        return *this;
    }


    RubiksCube &f() override 
    {
        this->rotateFace(2);

        char temp_arr[3] = {};
        for(int i=0;i<3;i++)
        {
            temp_arr[i] = cube[getIndex(0, 2, i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(0, 2, i)] = cube[getIndex(1, 2-i, 2)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(1, 2-i, 2)] = cube[getIndex(5, 0, 2-i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(5, 0, 2-i)] = cube[getIndex(3, i, 0)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(3, i, 0)] = temp_arr[i];
        }
        return *this;
    }

    RubiksCube &fPrime() override 
    {
        this->f();
        this->f();
        this->f();

        return *this;
    }

    RubiksCube &f2() override
    {
        this->f();
        this->f();

        return *this;
    }



    RubiksCube &r() override 
    {
        this->rotateFace(3);

        char temp_arr[3] = {};
        for(int i=0;i<3;i++)
        {
            temp_arr[i] = cube[getIndex(0, 2-i, 2)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(0, 2-i, 2)] = cube[getIndex(2, 2-i, 2)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(2, 2-i, 2)] = cube[getIndex(5, 2-i, 2)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(5, 2-i, 2)] = cube[getIndex(4, i, 0)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(4, i, 0)] = temp_arr[i];
        }
        return *this;
    }

    RubiksCube &rPrime() override 
    {
        this->r();
        this->r();
        this->r();

        return *this;
    }

    RubiksCube &r2() override
    {
        this->r();
        this->r();

        return *this;
    }



    RubiksCube &b() override 
    {
        this->rotateFace(4);

        char temp_arr[3] = {};
        for(int i=0;i<3;i++)
        {
            temp_arr[i] = cube[getIndex(0, 0, 2-i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(0, 0, 2-i)] = cube[getIndex(3, 2-i, 2)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(3, 2-i, 2)] = cube[getIndex(5, 2, i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(5, 2, i)] = cube[getIndex(1, i, 0)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(1, i, 0)] = temp_arr[i];
        }
        return *this;
    }

    RubiksCube &bPrime() override 
    {
        this->b();
        this->b();
        this->b();

        return *this;
    }

    RubiksCube &b2() override
    {
        this->b();
        this->b();

        return *this;
    }



    RubiksCube &d() override 
    {
        this->rotateFace(5);

        char temp_arr[3] = {};
        for(int i=0;i<3;i++)
        {
            temp_arr[i] = cube[getIndex(2, 2, i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(2, 2, i)] = cube[getIndex(1, 2, i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(1, 2, i)] = cube[getIndex(4, 2, i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(4, 2, i)] = cube[getIndex(3, 2, i)];
        }
        for(int i=0;i<3;i++)
        {
            cube[getIndex(3, 2, i)] = temp_arr[i];
        }
        return *this;
    }

    RubiksCube &dPrime() override 
    {
        this->d();
        this->d();
        this->d();

        return *this;
    }

    RubiksCube &d2() override
    {
        this->d();
        this->d();

        return *this;
    }


/*
The equality operator (==) checks if two RubiksCube1dArray objects have the same state.
The assignment operator (=) copies the state from one RubiksCube1dArray object to another.
*/

    bool operator==(const RubiksCube1dArray &r1) const 
    {
        for(int i=0;i<54;i++)
        {
            if(cube[i] != r1.cube[i])
            {
                return false;
            }
        }
        return true;
    }

    
    RubiksCube1dArray &operator=(const RubiksCube1dArray &r1)
    {
        for(int i=0;i<54;i++)
        {
            cube[i] = r1.cube[i];
        }
        return *this;
    }
};

/*
This struct defines a hash function for RubiksCube1dArray, converting its state to a string and then hashing the string using std::hash.
*/

struct Hash1d
{
    size_t operator()(const RubiksCube1dArray &r1) const
    {
        string str = "";
        for(int i=0;i<54;i++)
        {
            str += r1.cube[i];
        }
        return hash<string>()(str);
    }
};
