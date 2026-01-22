#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>

/*
########################################################################
c Copyright 2025 Predictive Science Inc.
c
c Licensed under the Apache License, Version 2.0 (the "License");
c you may not use this file except in compliance with the License.
c You may obtain a copy of the License at
c
c    http://www.apache.org/licenses/LICENSE-2.0
c
c Unless required by applicable law or agreed to in writing, software
c distributed under the License is distributed on an "AS IS" BASIS,
c WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
c implied.
c See the License for the specific language governing permissions and
c limitations under the License.
c#######################################################################
*/

cusparseHandle_t cusparseHandle=NULL;
cusparseStatus_t cusparseStatus;

cusparseSpMatDescr_t        L_mat;
cusparseSpMatDescr_t        U_mat;
cusparseDnVecDescr_t        DenseVecX;
cusparseDnVecDescr_t        DenseVecY;
cusparseMatDescr_t          M_described=0;
cusparseSpSVDescr_t         L_described=0;
cusparseSpSVDescr_t         U_described=0;
csrilu02Info_t              M_analyzed=0;

void * Mbuffer;
void * Lbuffer;
void * Ubuffer;

int Mbuf_size;
size_t Lbuf_size;
size_t Ubuf_size;

int struct_zero;
int n_zero;
int N_global;

const cusparseSolvePolicy_t M_policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t   L_trans  = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseOperation_t   U_trans  = CUSPARSE_OPERATION_NON_TRANSPOSE;

const double alpha = 1.0;

double* restrict x_64;
double* restrict y_64;


cusparseHandle_t cusparseHandle_pot2dh=NULL;
cusparseSpMatDescr_t        L_mat_pot2dh;
cusparseSpMatDescr_t        U_mat_pot2dh;
cusparseDnVecDescr_t        DenseVecX_pot2dh;
cusparseDnVecDescr_t        DenseVecY_pot2dh;
cusparseMatDescr_t          M_described_pot2dh=0;
cusparseSpSVDescr_t         L_described_pot2dh=0;
cusparseSpSVDescr_t         U_described_pot2dh=0;
csrilu02Info_t              M_analyzed_pot2dh=0;
void * Mbuffer_pot2dh;
void * Lbuffer_pot2dh;
void * Ubuffer_pot2dh;
int Mbuf_size_pot2dh;
int N_global_pot2dh;
double* restrict x_64_pot2dh;
double* restrict y_64_pot2dh;


cusparseHandle_t cusparseHandle_pot2d=NULL;
cusparseSpMatDescr_t        L_mat_pot2d;
cusparseSpMatDescr_t        U_mat_pot2d;
cusparseDnVecDescr_t        DenseVecX_pot2d;
cusparseDnVecDescr_t        DenseVecY_pot2d;
cusparseMatDescr_t          M_described_pot2d=0;
cusparseSpSVDescr_t         L_described_pot2d=0;
cusparseSpSVDescr_t         U_described_pot2d=0;
csrilu02Info_t              M_analyzed_pot2d=0;
void * Mbuffer_pot2d;
void * Lbuffer_pot2d;
void * Ubuffer_pot2d;
int Mbuf_size_pot2d;
int N_global_pot2d;
double* restrict x_64_pot2d;
double* restrict y_64_pot2d;

/* ******************************************************************* */
/* *** Initialize the cusparse functions in SP: *** */
/* ******************************************************************* */

void load_lusol_cusparse(double* restrict CSR_LU, int* restrict CSR_I,
                         int* restrict CSR_J, int N, int M)
{
  // If already initialized with the same N, skip reinitialization
  if (cusparseHandle != NULL && N_global == N) {
    return;
  }

  // Clean up any existing handles if N has changed
  if (cusparseHandle != NULL) {
    unload_lusol_cusparse();
  }

  N_global = N;

  // Allocate global scratch arrays and initialize to 0.
  cudaMalloc((void**)&x_64,N_global*sizeof(double));
  cudaMalloc((void**)&y_64,N_global*sizeof(double));
  cudaMemset((void*) x_64,0,N_global*sizeof(double));
  cudaMemset((void*) y_64,0,N_global*sizeof(double));

  // Setup cusparse.
  cusparseCreate(&cusparseHandle);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateMatDescr(&M_described);
  cusparseSetMatIndexBase(M_described, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatType(M_described, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseStatus = cusparseCreateCsrilu02Info(&M_analyzed);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Info Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer size for M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02_bufferSize(cusparseHandle, N, M,
                    M_described, CSR_LU, CSR_I, CSR_J,
                    M_analyzed, &Mbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Buffer Size Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((void**)&Mbuffer, Mbuf_size);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02_analysis(cusparseHandle,
                    N, M, M_described, CSR_LU, CSR_I, CSR_J,
                    M_analyzed, M_policy, Mbuffer);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle,
                    M_analyzed, &struct_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
      printf(" ERROR! A(%d,%d) is missing\n",
           struct_zero, struct_zero);
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set preconditioner (M=LU)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02(cusparseHandle, N, M, M_described,
                    CSR_LU, CSR_I, CSR_J, M_analyzed, M_policy, Mbuffer);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Formation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle,
                    M_analyzed, &n_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
     printf(" ERROR! M(%d,%d) is zero\n", n_zero, n_zero);
     exit(1);
  }

  cudaFree(Mbuffer);
  cusparseDestroyCsrilu02Info(M_analyzed);
  cusparseDestroyMatDescr(M_described);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Create the sparse matrix
  cusparseStatus = cusparseCreateCsr(&L_mat, N, N, M, CSR_I, CSR_J,
      CSR_LU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);
  if (cusparseStatus!=0){
      printf(" ERROR! L CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&L_described);

  // Set filling mode types
  cusparseFillMode_t L_fill_mode = CUSPARSE_FILL_MODE_LOWER;
  cusparseSpMatSetAttribute(L_mat, CUSPARSE_SPMAT_FILL_MODE,
      &L_fill_mode, sizeof(L_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t L_type_diag = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseSpMatSetAttribute(L_mat, CUSPARSE_SPMAT_DIAG_TYPE,
      &L_type_diag, sizeof(L_type_diag));

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up U
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Creat the sparse matrix
  cusparseStatus = cusparseCreateCsr(&U_mat, N, N, M, CSR_I, CSR_J,
      CSR_LU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);

  if (cusparseStatus!=0){
      printf(" ERROR! U CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&U_described);

  // Set filling mode types
  cusparseFillMode_t U_fill_mode = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(U_mat, CUSPARSE_SPMAT_FILL_MODE,
      &U_fill_mode, sizeof(U_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t U_type_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(U_mat, CUSPARSE_SPMAT_DIAG_TYPE,
      &U_type_diag, sizeof(U_type_diag));

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up Dense Vectors
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateDnVec(&DenseVecX, N, x_64, CUDA_R_64F);
  cusparseCreateDnVec(&DenseVecY, N, y_64, CUDA_R_64F);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer sizes
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle, L_trans,
        &alpha, L_mat, DenseVecX, DenseVecY, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described,&Lbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! L Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle, U_trans,
        &alpha, U_mat, DenseVecX, DenseVecY, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described,&Ubuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! U Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((void**)&Lbuffer, Lbuf_size);
  cudaMalloc((void**)&Ubuffer, Ubuf_size);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle, L_trans,
        &alpha, L_mat, DenseVecX, DenseVecY, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described, Lbuffer);
    if (cusparseStatus!=0){
      printf(" ERROR! L Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze U
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle, U_trans,
        &alpha, U_mat, DenseVecX, DenseVecY, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described, Ubuffer);
    if (cusparseStatus!=0){
      printf(" ERROR! U Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();
}


/* ******************************************************************* */
/* *** Do the Solve Phase: *** */
/* ******************************************************************* */

void lusol_cusparse(double* restrict x)
{
  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Update the Dense Vector (already linked to DenseVecX in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_64)
  for (int i=0;i<N_global;i++){
    x_64[i] = x[i];
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Forward solve (Ly=x) (DenseVecY already linked to y_64 pointer in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle, L_trans,
      &alpha, L_mat, DenseVecX, DenseVecY, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, L_described);
  if (cusparseStatus!=0){
      printf(" ERROR! Forward Solve Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Backward solve (Ux=y)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle, U_trans,
      &alpha, U_mat, DenseVecY, DenseVecX, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, U_described);
  if (cusparseStatus!=0){
      printf(" ERROR! Backward Solve Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Convert result to double precision.
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_64)
  for (int i=0;i<N_global;i++){
    x[i] = x_64[i];
  }

}

/* ******************************************************************* */
/* *** Free the Memory used by cusparse: *** */
/* ******************************************************************* */


void unload_lusol_cusparse()
{
  if (L_described != 0) {
    cusparseSpSV_destroyDescr(L_described);
    L_described = 0;
  }
  if (U_described != 0) {
    cusparseSpSV_destroyDescr(U_described);
    U_described = 0;
  }
  if (L_mat != NULL) {
    cusparseDestroySpMat(L_mat);
    L_mat = NULL;
  }
  if (U_mat != NULL) {
    cusparseDestroySpMat(U_mat);
    U_mat = NULL;
  }
  if (DenseVecX != NULL) {
    cusparseDestroyDnVec(DenseVecX);
    DenseVecX = NULL;
  }
  if (DenseVecY != NULL) {
    cusparseDestroyDnVec(DenseVecY);
    DenseVecY = NULL;
  }

  if (Lbuffer != NULL) {
    cudaFree(Lbuffer);
    Lbuffer = NULL;
  }
  if (Ubuffer != NULL) {
    cudaFree(Ubuffer);
    Ubuffer = NULL;
  }
  if (x_64 != NULL) {
    cudaFree(x_64);
    x_64 = NULL;
  }
  if (y_64 != NULL) {
    cudaFree(y_64);
    y_64 = NULL;
  }

  if (cusparseHandle != NULL) {
    cusparseDestroy(cusparseHandle);
    cusparseHandle = NULL;
  }

  cudaDeviceSynchronize();
}



/* ******************************************************************* */
/* *** Initialize the cusparse functions in SP: *** */
/* ******************************************************************* */

void load_lusol_cusparse_pot2dh(double* restrict CSR_LU, int* restrict CSR_I,
                         int* restrict CSR_J,int N, int M)
{
  N_global_pot2dh = N;

  // Allocate global scratch arrays and initialize to 0.
  cudaMalloc((void**)&x_64_pot2dh,N_global_pot2dh*sizeof(double));
  cudaMalloc((void**)&y_64_pot2dh,N_global_pot2dh*sizeof(double));
  cudaMemset((void*) x_64_pot2dh,0,N_global_pot2dh*sizeof(double));
  cudaMemset((void*) y_64_pot2dh,0,N_global_pot2dh*sizeof(double));

  // Setup cusparse.
  cusparseCreate(&cusparseHandle_pot2dh);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateMatDescr(&M_described_pot2dh);
  cusparseSetMatIndexBase(M_described_pot2dh, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatType(M_described_pot2dh, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseStatus = cusparseCreateCsrilu02Info(&M_analyzed_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Info Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer size for M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02_bufferSize(cusparseHandle_pot2dh, N, M,
                    M_described_pot2dh, CSR_LU, CSR_I, CSR_J,
                    M_analyzed_pot2dh, &Mbuf_size_pot2dh);

  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Buffer Size Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((void**)&Mbuffer_pot2dh, Mbuf_size_pot2dh);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02_analysis(cusparseHandle_pot2dh,
                    N, M, M_described_pot2dh, CSR_LU, CSR_I, CSR_J,
              M_analyzed_pot2dh, M_policy, Mbuffer_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2dh,
                    M_analyzed_pot2dh, &struct_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
      printf(" ERROR! A(%d,%d) is missing\n",
           struct_zero, struct_zero);
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set preconditioner (M=LU)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02(cusparseHandle_pot2dh, N, M, M_described_pot2dh,
                    CSR_LU, CSR_I, CSR_J, M_analyzed_pot2dh, M_policy, Mbuffer_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Formation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2dh,
                    M_analyzed_pot2dh, &n_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
     printf(" ERROR! M(%d,%d) is zero\n", n_zero, n_zero);
     exit(1);
  }

  cudaFree(Mbuffer_pot2dh);
  cusparseDestroyCsrilu02Info(M_analyzed_pot2dh);
  cusparseDestroyMatDescr(M_described_pot2dh);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Create the sparse matrix
  cusparseStatus = cusparseCreateCsr(&L_mat_pot2dh, N, N, M, CSR_I, CSR_J,
      CSR_LU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);
  if (cusparseStatus!=0){
      printf(" ERROR! L CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&L_described_pot2dh);

  // Set filling mode types
  cusparseFillMode_t L_fill_mode = CUSPARSE_FILL_MODE_LOWER;
  cusparseSpMatSetAttribute(L_mat_pot2dh, CUSPARSE_SPMAT_FILL_MODE,
      &L_fill_mode, sizeof(L_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t L_type_diag = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseSpMatSetAttribute(L_mat_pot2dh, CUSPARSE_SPMAT_DIAG_TYPE,
      &L_type_diag, sizeof(L_type_diag));

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up U
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Creat the sparse matrix
  cusparseStatus = cusparseCreateCsr(&U_mat_pot2dh, N, N, M, CSR_I, CSR_J,
      CSR_LU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);

  if (cusparseStatus!=0){
      printf(" ERROR! U CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&U_described_pot2dh);

  // Set filling mode types
  cusparseFillMode_t U_fill_mode = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(U_mat_pot2dh, CUSPARSE_SPMAT_FILL_MODE,
      &U_fill_mode, sizeof(U_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t U_type_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(U_mat_pot2dh, CUSPARSE_SPMAT_DIAG_TYPE,
      &U_type_diag, sizeof(U_type_diag));

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up Dense Vectors
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateDnVec(&DenseVecX_pot2dh, N, x_64_pot2dh, CUDA_R_64F);
  cusparseCreateDnVec(&DenseVecY_pot2dh, N, y_64_pot2dh, CUDA_R_64F);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer sizes
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2dh, L_trans,
        &alpha, L_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2dh,&Lbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! L Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2dh, U_trans,
        &alpha, U_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2dh,&Ubuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! U Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((void**)&Lbuffer_pot2dh, Lbuf_size);
  cudaMalloc((void**)&Ubuffer_pot2dh, Ubuf_size);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2dh, L_trans,
        &alpha, L_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2dh, Lbuffer_pot2dh);
    if (cusparseStatus!=0){
      printf(" ERROR! L Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze U
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2dh, U_trans,
        &alpha, U_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2dh, Ubuffer_pot2dh);
    if (cusparseStatus!=0){
      printf(" ERROR! U Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();
}


/* ******************************************************************* */
/* *** Do the Solve Phase: *** */
/* ******************************************************************* */

void lusol_cusparse_pot2dh(double* restrict x)
{
  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Update the Dense Vector (already linked to DenseVecX_pot2dh in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_64_pot2dh)
  for (int i=0;i<N_global_pot2dh;i++){
    x_64_pot2dh[i] = x[i];
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Forward solve (Ly=x) (DenseVecY_pot2dh already linked to y_64_pot2dh pointer in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2dh, L_trans,
      &alpha, L_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! Forward Solve Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Backward solve (Ux=y)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2dh, U_trans,
      &alpha, U_mat_pot2dh, DenseVecY_pot2dh, DenseVecX_pot2dh, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! Backward Solve Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Convert result to double precision.
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_64_pot2dh)
  for (int i=0;i<N_global_pot2dh;i++){
    x[i] = x_64_pot2dh[i];
  }

}

/* ******************************************************************* */
/* *** Initialize the cusparse functions in SP: *** */
/* ******************************************************************* */

void load_lusol_cusparse_pot2d(double* restrict CSR_LU, int* restrict CSR_I,
                         int* restrict CSR_J,int N, int M)
{
  N_global_pot2d = N;

  // Allocate global scratch arrays and initialize to 0.
  cudaMalloc((void**)&x_64_pot2d,N_global_pot2d*sizeof(double));
  cudaMalloc((void**)&y_64_pot2d,N_global_pot2d*sizeof(double));
  cudaMemset((void*) x_64_pot2d,0,N_global_pot2d*sizeof(double));
  cudaMemset((void*) y_64_pot2d,0,N_global_pot2d*sizeof(double));

  // Setup cusparse.
  cusparseCreate(&cusparseHandle_pot2d);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateMatDescr(&M_described_pot2d);
  cusparseSetMatIndexBase(M_described_pot2d, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatType(M_described_pot2d, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseStatus = cusparseCreateCsrilu02Info(&M_analyzed_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Info Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer size for M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02_bufferSize(cusparseHandle_pot2d, N, M,
                    M_described_pot2d, CSR_LU, CSR_I, CSR_J,
                    M_analyzed_pot2d, &Mbuf_size_pot2d);

  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Buffer Size Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((void**)&Mbuffer_pot2d, Mbuf_size_pot2d);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02_analysis(cusparseHandle_pot2d,
                    N, M, M_described_pot2d, CSR_LU, CSR_I, CSR_J,
              M_analyzed_pot2d, M_policy, Mbuffer_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2d,
                    M_analyzed_pot2d, &struct_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
      printf(" ERROR! A(%d,%d) is missing\n",
           struct_zero, struct_zero);
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set preconditioner (M=LU)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseDcsrilu02(cusparseHandle_pot2d, N, M, M_described_pot2d,
                    CSR_LU, CSR_I, CSR_J, M_analyzed_pot2d, M_policy, Mbuffer_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Formation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2d,
                    M_analyzed_pot2d, &n_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
     printf(" ERROR! M(%d,%d) is zero\n", n_zero, n_zero);
     exit(1);
  }

  cudaFree(Mbuffer_pot2d);
  cusparseDestroyCsrilu02Info(M_analyzed_pot2d);
  cusparseDestroyMatDescr(M_described_pot2d);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Create the sparse matrix
  cusparseStatus = cusparseCreateCsr(&L_mat_pot2d, N, N, M, CSR_I, CSR_J,
      CSR_LU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);
  if (cusparseStatus!=0){
      printf(" ERROR! L CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&L_described_pot2d);

  // Set filling mode types
  cusparseFillMode_t L_fill_mode = CUSPARSE_FILL_MODE_LOWER;
  cusparseSpMatSetAttribute(L_mat_pot2d, CUSPARSE_SPMAT_FILL_MODE,
      &L_fill_mode, sizeof(L_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t L_type_diag = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseSpMatSetAttribute(L_mat_pot2d, CUSPARSE_SPMAT_DIAG_TYPE,
      &L_type_diag, sizeof(L_type_diag));

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up U
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Creat the sparse matrix
  cusparseStatus = cusparseCreateCsr(&U_mat_pot2d, N, N, M, CSR_I, CSR_J,
      CSR_LU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);

  if (cusparseStatus!=0){
      printf(" ERROR! U CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&U_described_pot2d);

  // Set filling mode types
  cusparseFillMode_t U_fill_mode = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(U_mat_pot2d, CUSPARSE_SPMAT_FILL_MODE,
      &U_fill_mode, sizeof(U_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t U_type_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(U_mat_pot2d, CUSPARSE_SPMAT_DIAG_TYPE,
      &U_type_diag, sizeof(U_type_diag));

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up Dense Vectors
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateDnVec(&DenseVecX_pot2d, N, x_64_pot2d, CUDA_R_64F);
  cusparseCreateDnVec(&DenseVecY_pot2d, N, y_64_pot2d, CUDA_R_64F);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer sizes
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2d, L_trans,
        &alpha, L_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2d,&Lbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! L Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2d, U_trans,
        &alpha, U_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2d,&Ubuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! U Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((void**)&Lbuffer_pot2d, Lbuf_size);
  cudaMalloc((void**)&Ubuffer_pot2d, Ubuf_size);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2d, L_trans,
        &alpha, L_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2d, Lbuffer_pot2d);
    if (cusparseStatus!=0){
      printf(" ERROR! L Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze U
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2d, U_trans,
        &alpha, U_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2d, Ubuffer_pot2d);
    if (cusparseStatus!=0){
      printf(" ERROR! U Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();
}


/* ******************************************************************* */
/* *** Do the Solve Phase: *** */
/* ******************************************************************* */

void lusol_cusparse_pot2d(double* restrict x)
{
  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Update the Dense Vector (already linked to DenseVecX_pot2d in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_64_pot2d)
  for (int i=0;i<N_global_pot2d;i++){
    x_64_pot2d[i] = x[i];
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Forward solve (Ly=x) (DenseVecY_pot2d already linked to y_64_pot2d pointer in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2d, L_trans,
      &alpha, L_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! Forward Solve Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Backward solve (Ux=y)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2d, U_trans,
      &alpha, U_mat_pot2d, DenseVecY_pot2d, DenseVecX_pot2d, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! Backward Solve Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Convert result to double precision.
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_64_pot2d)
  for (int i=0;i<N_global_pot2d;i++){
    x[i] = x_64_pot2d[i];
  }

}
