#include <stdio.h>

int offset(int r, int c, int ch, int C, int num_chs)
{
    int ret = ch+num_chs*(c+C*r);
    return ret;
}



float resample_bilinear(float* pixbuf, float rem_row, float rem_col, int r0, int r1, int c0, int c1, int ch, int C, int num_chs)
{
    float v00 = pixbuf[offset(r0, c0, ch, C, num_chs)];
    float v01 = pixbuf[offset(r0, c1, ch, C, num_chs)];
    float v10 = pixbuf[offset(r1, c0, ch, C, num_chs)];
    float v11 = pixbuf[offset(r1, c1, ch, C, num_chs)];
    float v0 = v00+(v01-v00)*rem_col;
    float v1 = v10+(v11-v10)*rem_col;
    float v = v0+(v1-v0)*rem_row;
    return v;
}

void interp_bilinear(float* pixbuf, int R, int C, int ch, float* sample_r, float* sample_c, float* retbuf, int num_samplings)
{
    int Rm = R-1;
    int Cm = C-1;
    int Rm2 = 2*(R-1);
    int Cm2 = 2*(C-1);
    for(int i=0; i<num_samplings; ++i)
    {
        int retbuf_offset = i*ch;

        float cur_row = sample_r[i];
        int irow;
        float rem_row;
        int odd_wraps_row;
        if(cur_row < 0)
        {
            cur_row = -cur_row;
        }
        irow = (int) cur_row;
        rem_row = cur_row - irow;
        int irow1 = irow+1;
        irow = irow%Rm2;
        irow = irow < R? irow:Rm2-irow;
        irow1 = irow1%Rm2;
        irow1 = irow1 < R? irow1:Rm2-irow1;

        float cur_col = sample_c[i];
        if(cur_col < 0)
        {
            cur_col = -cur_col;
        }
        int icol = (int) cur_col;
        float rem_col = cur_col-icol;
        int icol1 = icol+1;
        icol=icol%Cm2;
        icol=icol<C?icol:Cm2-icol;
        icol1=icol1%Cm2;
        icol1=icol1 < C?icol1:Cm2-icol1;
        for(int cur_ch=0; cur_ch<ch; ++cur_ch)
        {
            retbuf[retbuf_offset+cur_ch] = resample_bilinear(pixbuf, rem_row, rem_col, irow, irow1, icol, icol1, cur_ch, C, ch);
        }
    }
}

int offset_nearest(int r, int c, int C)
{
    int ret = c+C*r;
    return ret;
}

// assume char is 8 bits
char resample_nearest(char* pixbuf, float rem_row, float rem_col, int r0, int r1, int c0, int c1, int C)
{
    int row = (rem_row > 0.5) ? r1: r0;
    int col = (rem_col > 0.5) ? c1: c0;
    float v = pixbuf[offset_nearest(row, col, C)];
    return v;
}

// duplicated code, but since the resample function is called so often
// so inlining could potentially lead to large speedups
// Inlining can not be done if the function call is an input void pointer
// C++ templates would be preferable, but C does not have those
// assuming a char is 8 bits
void interp_nearest(char* pixbuf, int R, int C, float* sample_r, float* sample_c, char* retbuf, int num_samplings)
{
    int Rm = R-1;
    int Cm = C-1;
    int Rm2 = 2*(R-1);
    int Cm2 = 2*(C-1);
    for(int i=0; i<num_samplings; ++i)
    {
        int retbuf_offset = i;

        float cur_row = sample_r[i];
        int irow;
        float rem_row;
        int odd_wraps_row;
        if(cur_row < 0)
        {
            cur_row = -cur_row;
        }
        irow = (int) cur_row;
        rem_row = cur_row - irow;
        int irow1 = irow+1;
        irow = irow%Rm2;
        irow = irow < R? irow:Rm2-irow;
        irow1 = irow1%Rm2;
        irow1 = irow1 < R? irow1:Rm2-irow1;

        float cur_col = sample_c[i];
        if(cur_col < 0)
        {
            cur_col = -cur_col;
        }
        int icol = (int) cur_col;
        float rem_col = cur_col-icol;
        int icol1 = icol+1;
        icol=icol%Cm2;
        icol=icol<C?icol:Cm2-icol;
        icol1=icol1%Cm2;
        icol1=icol1 < C?icol1:Cm2-icol1;
        retbuf[retbuf_offset] = resample_nearest(pixbuf, rem_row, rem_col, irow, irow1, icol, icol1, C);
    }
}

