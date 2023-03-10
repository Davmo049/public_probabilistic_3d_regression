#include <stdio.h>

void yrot(int* square)
{
    int ymin = square[1];
    int minidx = 0;
    for(int i=0; i < 4; ++i)
    {
        int cand = square[i*2+1];
        if (ymin > cand)
        {
            ymin = cand;
            minidx = i;
        }
    }
    int tmpx;
    int tmpy;
    switch(minidx)
    {
        case 0:
            break;
        case 1:
            tmpx = square[0];
            tmpy = square[1];
            for(int j=0; j<6; ++j)
            {
                square[j] = square[j+2];
            }
            square[6] = tmpx;
            square[7] = tmpy;
            break;
        case 2:
            tmpx = square[0];
            tmpy = square[1];
            square[0] = square[4];
            square[1] = square[5];
            square[4] = tmpx;
            square[5] = tmpy;
            tmpx = square[2];
            tmpy = square[3];
            square[2] = square[6];
            square[3] = square[7];
            square[6] = tmpx;
            square[7] = tmpy;
            break;
        case 3:
            tmpx = square[6];
            tmpy = square[7];
            for(int j=7; j>1; --j)
            {
                square[j] = square[j-2];
            }
            square[0] = tmpx;
            square[1] = tmpy;
            break;
    }
}

typedef struct
{
    int slx;
    int sly;
    int srx;
    int sry;
    int elx;
    int ely;
    int erx;
    int ery;
} Rombous;

typedef struct
{
    float* pixbuf;
    int R;
    int C;
} Image;

void drawrom(Image* im, Rombous* rombous, int starty, int endy, float* color)
{
    for(int y=starty; y < endy; ++y)
    {
        int startx = rombous->slx+((y-rombous->sly)*(rombous->elx-rombous->slx))/(rombous->ely-rombous->sly);
        int endx = rombous->srx+((y-rombous->sry)*(rombous->erx-rombous->srx))/(rombous->ery-rombous->sry);
        if (startx < 0)
        {
            startx = 0;
        }
        if (endx > im->C)
        {
            endx = im->C;
        }
        for(int x=startx; x < endx; ++x)
        {
            im->pixbuf[0+3*(x+im->C*y)] = color[0];
            im->pixbuf[1+3*(x+im->C*y)] = color[1];
            im->pixbuf[2+3*(x+im->C*y)] = color[2];
        }
    }
}

void fill(float* pixbuf, int R, int C, int* square, float* color)
{
    yrot(square);
    Image im;
    im.pixbuf = pixbuf;
    im.R = R;
    im.C = C;
    Rombous drawromb;
    drawromb.slx = square[0];
    drawromb.sly = square[1];
    drawromb.srx = square[0];
    drawromb.sry = square[1];
    drawromb.elx = square[2];
    drawromb.ely = square[3];
    drawromb.erx = square[6];
    drawromb.ery = square[7];
    int leftidx = 3;
    int rightidx = 7;
    int starty = square[1];
    while (1)
    {
        int flag = square[leftidx] < square[rightidx];
        if (flag)
        {
            int endy = square[leftidx];
            drawrom(&im, &drawromb, starty, endy, color);
            starty = endy;
        }
        else
        {
            int endy = square[rightidx];
            drawrom(&im, &drawromb, starty, endy, color);
            starty = endy;
        }
        if (leftidx == rightidx)
        {
            break;
        }
        if (flag)
        {
            leftidx += 2;
            drawromb.slx = drawromb.elx;
            drawromb.sly = drawromb.ely;
            drawromb.elx = square[leftidx-1];
            drawromb.ely = square[leftidx];
        }
        else
        {
            rightidx -= 2;
            drawromb.srx = drawromb.erx;
            drawromb.sry = drawromb.ery;
            drawromb.erx = square[rightidx-1];
            drawromb.ery = square[rightidx];
        }
    }
}
