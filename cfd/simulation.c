#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"
#include <mpi.h>

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int *ileft, *iright;
extern int nprocs, proc;  //
extern MPI_Comm comm;
extern int dim[2], period[2], reorder;
extern int coord[2], id;
extern int unitx, unity, remainx, remainy;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re)
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                    gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                    (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                    gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                    /(4.0*delx);
                duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                    gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                    (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                    gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                    /(4.0*dely);
                laplu = (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx+
                    (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely;
   
                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                    gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                    (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                    gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                    /(4.0*delx);
                dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                    gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                    (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                    gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                    /(4.0*dely);

                laplv = (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx+
                    (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely;

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    /* f & g at external boundaries */
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (i=1; i<=imax; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely)
{
    int i, j;

    for (i=1;i<=imax;i++) {
        for (j=1;j<=jmax;j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = (
                             (f[i][j]-f[i-1][j])/delx +
                             (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
            }
        }
    }
}

int do_decomposition(int imax, int jmax)
{
    //calculate the size of matrix for each process
    unitx = floor(jmax/dim[0]);
    unity = floor(imax/dim[1]);

    //if size can't be divided into int, put rest of data into final part
    remainx = unitx + jmax-(unitx*dim[0]);
    remainy = unity + imax-(unity*dim[1]);   
    printf("do_decomposition.\n");
    return 0;
}

int cartsian_topology(int imax, int jmax)
{
    if(imax > jmax)
    {
        int a = (int)ceil(sqrt(nprocs*imax/jmax));
        dim[1] = min(a, nprocs);
        dim[0] = nprocs/dim[1];
        period[0] = 0;
        period[1] = 0;
        reorder = 0;
    }
    else
    {   
        int a = (int)ceil(sqrt(nprocs*imax/jmax));
        dim[0] = min(a, nprocs);
        dim[1] = nprocs/dim[0];
        period[0] = 0;
        period[1] = 0;
        reorder = 0;   
    }

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);
    printf("cartsian_topology, finished");
    return 0;

}

int par_poisson(float **local_p, float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull)
{
    int up, down, left, right;
    MPI_Comm_rank(comm, &proc);
    MPI_Cart_shift(comm, 0, 1, &right, &left);
    MPI_Cart_shift(comm, 1, 1, &down, &up);

    //printf("%d,%d,%d,%d",up, down, left, right);

    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;
    float local_p0 = 0.0;
    
    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    int my_coords[2];
    int cart_coords[2];
    MPI_Cart_coords(comm, proc, 2, cart_coords);
    my_coords[0] = cart_coords[0];
    my_coords[1] = dim[1]-cart_coords[1]-1;


    int offsetx = (my_coords[0]<(dim[0]-1)) ? unitx : remainx;
    int offsety = (my_coords[1]<(dim[1]-1)) ? unity : remainy;

    /* Calculate sum of squares */ //can be omp
    for (i = 1+my_coords[1]*unity; i < 1+my_coords[1]*unity+offsety; i++) {
         for (j = 1+ my_coords[0]*unitx; j < 1+my_coords[0]*unitx+offsetx; j++) {
            if (flag[i][j] & C_F) { local_p0 += p[i][j]*p[i][j]; }
        }
    }
    MPI_Allreduce(&local_p0, &p0, 1, MPI_FLOAT, MPI_SUM, comm);

    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }
    //Initial local p
    for(int m=0; m<=imax+1; m++){
        for(int n=0; n<=jmax+1; n++){
            local_p[m][n]= 0;
        }
    }
    for (i = my_coords[1]*unity; i <= 1+my_coords[1]*unity+offsety; i++) {
         for (j = my_coords[0]*unitx; j <= 1+my_coords[0]*unitx+offsetx; j++) {
            local_p[i][j] = p[i][j];
        }
    }

    /* Red/Black SOR-iteration */
    //loop controll iteration
    for (iter = 0; iter < itermax; iter++) {
        //loop contoll red & black
        for (rb = 0; rb <= 1; rb++) {
            for (i = 1+my_coords[1]*unity; i < 1+my_coords[1]*unity+offsety; i++) {
                // y direction
                for (j = 1+ my_coords[0]*unitx; j < 1+my_coords[0]*unitx+offsetx; j++) {
                    //skip odd or equal
                    if ((i+j) % 2 != rb) { continue; }
                    //C_F fluid cell,B_NSEW obstacle cell
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        local_p[i][j] = (1.-omega)*local_p[i][j] - 
                              beta_2*(
                                    (local_p[i+1][j]+local_p[i-1][j])*rdx2
                                  + (local_p[i][j+1]+local_p[i][j-1])*rdy2
                                  -  rhs[i][j]
                              );
                        // if(local_p[i][j]!=0 && rb ==0){
                        //     printf("%f(%d) ", local_p[i][j], proc);
                        // }
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        local_p[i][j] = (1.-omega)*local_p[i][j] -
                            beta_mod*(
                                  (eps_E*local_p[i+1][j]+eps_W*local_p[i-1][j])*rdx2
                                + (eps_N*local_p[i][j+1]+eps_S*local_p[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                        // if(local_p[i][j]!=0 && rb ==0){
                        //      printf("%f(%d) ", local_p[i][j], proc);
                        // }
                    }
                } /* end of j */
            } /* end of i */

            if(up!=-1){
                float *buffer = (float*)malloc(sizeof(float)*offsetx);
                float *recv_buffer =  (float*)malloc(sizeof(float)*offsetx);
                int start = 1+my_coords[0]*unitx;
                int end = start+offsetx;
                for (j = start; j < end; j++){
                    buffer[j-start] = local_p[1+my_coords[0]*unity+offsety][j];
                }
                MPI_Sendrecv(buffer, offsetx, MPI_FLOAT, up, 0,
                        recv_buffer, offsetx, MPI_FLOAT, up, 2,
                         comm, MPI_STATUS_IGNORE);

                for (j = start; j < end; j++){
                    local_p[my_coords[0]*unity+offsety][j] = recv_buffer[j-start];
                 }
                free(buffer);
                free(recv_buffer);
                //printf("%d,(%d,%d)\n",proc, my_coords[0],my_coords[1]);
            }
             

            //communicate down
            if(down!=-1){
                float *buffer = (float*)malloc(sizeof(float)*offsetx);
                float *recv_buffer =  (float*)malloc(sizeof(float)*offsetx);
                int start = 1 + my_coords[0]*unitx;
                int end = start+ offsetx;
                for (j =start; j < end; j++){
                    buffer[j-start] = local_p[my_coords[0]*unity][j];
                }

                MPI_Sendrecv(buffer, offsetx, MPI_FLOAT, down, 2,
                         recv_buffer, offsetx, MPI_FLOAT, down, 0,
                         comm, MPI_STATUS_IGNORE);
                for (j = start; j < end; j++){
                    local_p[my_coords[0]*unity+1][j] = recv_buffer[j-start];
                }
                free(buffer);
                free(recv_buffer);
                //printf("%d\n",proc);
            } 

            if(left!=-1){
                float *buffer = (float*)malloc(sizeof(float)*offsety);
                float *recv_buffer =  (float*)malloc(sizeof(float)*offsety);
                int start = 1+my_coords[1]*unity;
                int end = start + offsety;
                for (j = start; j < end; j++){
                    buffer[j-start] = local_p[my_coords[1]*unitx+1][j];
                }
                MPI_Sendrecv(buffer, offsety, MPI_FLOAT, left, 1,
                         recv_buffer, offsety, MPI_FLOAT, left, 3,
                         comm, MPI_STATUS_IGNORE);
                for (j = start; j < end; j++){
                    local_p[my_coords[1]*unitx][j] = recv_buffer[j-start];
                }
                free(buffer);
                free(recv_buffer);
                //printf("%d,(%d,%d)\n",proc, my_coords[0],my_coords[1]);
            } 

            if(right!=-1){
                float *buffer = (float*)malloc(sizeof(float)*offsety);
                float *recv_buffer =  (float*)malloc(sizeof(float)*offsety);
                int start = 1+my_coords[1]*unity;
                int end = start + offsety;
                for (j =start; j < end; j++){
                     buffer[j-start] = local_p[my_coords[1]*unitx+offsetx][j];
                }
                MPI_Sendrecv(buffer, offsety, MPI_FLOAT, right, 3,
                         recv_buffer, offsety, MPI_FLOAT, right, 1,
                         comm, MPI_STATUS_IGNORE);
                for (j = start;  j < end; j++){
                    local_p[my_coords[1]*unitx+offsetx+1][j] = recv_buffer[j-start];
                }
                free(buffer);
                free(recv_buffer);
                //printf("%d,(%d,%d)\n",proc, my_coords[0],my_coords[1]);
            }
        }


        MPI_Barrier(comm);
        /*Partial computation of residual*/
        *res = 0.0;
        float local_res = 0.0;
        for (i = 1+my_coords[1]*unity; i < 1+my_coords[1]*unity+offsety; i++) {
            // y direction
             for (j = 1+ my_coords[0]*unitx; j < 1+my_coords[0]*unitx+offsetx; j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells*/
                    add = (eps_E*(local_p[i+1][j]-local_p[i][j]) - 
                        eps_W*(local_p[i][j]-local_p[i-1][j])) * rdx2  +
                        (eps_N*(local_p[i][j+1]-local_p[i][j]) -
                        eps_S*(local_p[i][j]-local_p[i][j-1])) * rdy2  -  rhs[i][j];
                    local_res += add*add;
                }
            }
        }
        
        MPI_Allreduce(&local_res, res, 1, MPI_FLOAT, MPI_SUM, comm);
        *res = sqrt((*res)/ifull)/p0;

        if (*res<eps) {
            break;
        }
    }   

    //clean boader
    for (i =0; i <= imax+1; i++) {
        for (j = 0; j <= jmax+1; j++) {
            if(i<(1+my_coords[1]*unity)||i >= 1+my_coords[1]*unity+offsety||j<(1+ my_coords[0]*unitx)||j>=(1+my_coords[0]*unitx+offsetx))
            {
                local_p[i][j]=0;
            }
        }
    }

    MPI_Barrier(comm);
    for(i = 0; i<=imax+1; i++)
    {
        MPI_Allreduce(local_p[i], p[i], jmax+2, MPI_FLOAT, MPI_SUM, comm);
    }

    if(proc == 0){
        // for(i=0;i<imax+2;i++){
        //     int count =0;
        //     for(j=0;j<jmax+2;j++){
        //         if(local_p[i][j]!=0){
        //             printf("%f(%d,%d)", local_p[i][j],i,j);
        //             count++;
        //         }
        //     }
        //     if(count>0)
        //     printf("\n");
        // }
        // printf("i-1%f,", p[515][9]);
        // printf("i+1%f,", p[517][9]);
        // printf("j-1%f,", p[516][8]);
        // printf("j+1%f,", p[516][10]);
        // printf("ij%f,", p[516][9]);
        printf("%f\n",*res);
    }
    return iter;
}

/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull)
{
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;
    
    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    /* Calculate sum of squares */
    for (i = 1; i <= imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }
   
    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    //loop controll iteration
    for (iter = 0; iter < itermax; iter++) {
        //loop contoll red & black
        for (rb = 0; rb <= 1; rb++) {
            // x direction
            for (i = 1; i <= imax; i++) {
                // y direction
                for (j = 1; j <= jmax; j++) {
                    //skip odd or equal
                    if ((i+j) % 2 != rb) { continue; }
                    //C_F fluid cell,B_NSEW obstacle cell
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1.-omega)*p[i][j] - 
                              beta_2*(
                                    (p[i+1][j]+p[i-1][j])*rdx2
                                  + (p[i][j+1]+p[i][j-1])*rdy2
                                  -  rhs[i][j]
                              );
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p[i][j] = (1.-omega)*p[i][j] -
                            beta_mod*(
                                  (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                                + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                        //printf("%f\n", rhs[i][j]);
                    }
                } /* end of j */
            } /* end of i */
        } /* end of rb */
        
        /* Partial computation of residual */
        *res = 0.0;
        for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) - 
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    *res += add*add;
                }
            }
        }
        *res = sqrt((*res)/ifull)/p0;
        /* convergence? */
        if (*res<eps) break;
    } /* end of iter */

    if(proc == 0)
    printf("%f,%d\n",*res, iter);
    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely)
{
    int i, j;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
        }
    }
}


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau)
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe; 

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        umax = 1.0e-10;
        vmax = 1.0e-10; 
        for (i=0; i<=imax+1; i++) {
            for (j=1; j<=jmax+1; j++) {
                umax = max(fabs(u[i][j]), umax);
            }
        }
        for (i=1; i<=imax+1; i++) {
            for (j=0; j<=jmax+1; j++) {
                vmax = max(fabs(v[i][j]), vmax);
            }
        }

        deltu = delx/umax;
        deltv = dely/vmax; 
        deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

        if (deltu<deltv) {
            *del_t = min(deltu, deltRe);
        } else {
            *del_t = min(deltv, deltRe);
        }
        *del_t = tau * (*del_t); /* multiply by safety factor */
    }
}
