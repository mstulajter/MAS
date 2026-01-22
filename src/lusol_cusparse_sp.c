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

cusparseHandle_t cusparseHandle_sp=NULL;
cusparseStatus_t cusparseStatus;

cusparseSpMatDescr_t        L_mat;
cusparseSpMatDescr_t        U_mat;
cusparseDnVecDescr_t        DenseVecX;
cusparseDnVecDescr_t        DenseVecY;
cusparseMatDescr_t          M_described_sp=0;
cusparseSpSVDescr_t         L_described_sp=0;
cusparseSpSVDescr_t         U_described_sp=0;
csrilu02Info_t              M_analyzed_sp=0;

void * Mbuffer;
void * Lbuffer;
void * Ubuffer;

int Mbuf_size;
size_t Lbuf_size;
size_t Ubuf_size;

int struct_zero;
int n_zero;
int N_global;
int M_global;

const cusparseSolvePolicy_t M_policy_sp = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t  L_trans_sp  = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseOperation_t  U_trans_sp  = CUSPARSE_OPERATION_NON_TRANSPOSE;

const float alpha_sp = 1.0f;

float* restrict x_32;
float* restrict y_32;
float* restrict CSR_LU_32;

// Timing statistics for load_lusol_cusparse_sp
float time_mem_alloc_sp = 0.0f;
float time_cusparse_create_sp = 0.0f;
float time_M_setup_sp = 0.0f;
float time_M_buffer_size_sp = 0.0f;
float time_M_buffer_alloc_sp = 0.0f;
float time_M_analysis_sp = 0.0f;
float time_M_preconditioner_sp = 0.0f;
float time_M_cleanup_sp = 0.0f;
float time_L_setup_sp = 0.0f;
float time_U_setup_sp = 0.0f;
float time_dense_vec_setup_sp = 0.0f;
float time_buffer_size_sp = 0.0f;
float time_buffer_alloc_sp = 0.0f;
float time_L_analysis_sp = 0.0f;
float time_U_analysis_sp = 0.0f;

cusparseHandle_t cusparseHandle_pot2dh_sp=NULL;
cusparseSpMatDescr_t        L_mat_pot2dh;
cusparseSpMatDescr_t        U_mat_pot2dh;
cusparseDnVecDescr_t        DenseVecX_pot2dh;
cusparseDnVecDescr_t        DenseVecY_pot2dh;
cusparseMatDescr_t          M_described_pot2dh_sp=0;
cusparseSpSVDescr_t         L_described_pot2dh_sp=0;
cusparseSpSVDescr_t         U_described_pot2dh_sp=0;
csrilu02Info_t              M_analyzed_pot2dh_sp=0;
void * Mbuffer_pot2dh;
void * Lbuffer_pot2dh;
void * Ubuffer_pot2dh;
int Mbuf_size_pot2dh;
int N_global_pot2dh;
float* restrict x_32_pot2dh;
float* restrict y_32_pot2dh;
float* restrict CSR_LU_32_pot2dh;


cusparseHandle_t cusparseHandle_pot2d_sp=NULL;
cusparseSpMatDescr_t        L_mat_pot2d;
cusparseSpMatDescr_t        U_mat_pot2d;
cusparseDnVecDescr_t        DenseVecX_pot2d;
cusparseDnVecDescr_t        DenseVecY_pot2d;
cusparseMatDescr_t          M_described_pot2d_sp=0;
cusparseSpSVDescr_t         L_described_pot2d_sp=0;
cusparseSpSVDescr_t         U_described_pot2d_sp=0;
csrilu02Info_t              M_analyzed_pot2d_sp=0;
void * Mbuffer_pot2d;
void * Lbuffer_pot2d;
void * Ubuffer_pot2d;
int Mbuf_size_pot2d;
int N_global_pot2d;
float* restrict x_32_pot2d;
float* restrict y_32_pot2d;
float* restrict CSR_LU_32_pot2d;

/* ******************************************************************* */
/* *** Initialize the cusparse functions in SP: *** */
/* ******************************************************************* */

void load_lusol_cusparse_sp(double* restrict CSR_LU, int* restrict CSR_I, 
                         int* restrict CSR_J, int N, int M)
{
  // If already initialized with the same N, skip reinitialization
  if (cusparseHandle_sp != NULL && N_global == N) {
    return;
  }

  // Clean up any existing handles if N has changed
  if (cusparseHandle_sp != NULL) {
    unload_lusol_cusparse_sp();
  }

  // Timing variables
  cudaEvent_t start, stop;
  float elapsed_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  N_global = N;
  M_global = M;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Memory allocation and initialization
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  // Allocate global scratch arrays and initialize to 0.
  cudaMalloc((void**)&x_32,N_global*sizeof(float));
  cudaMalloc((void**)&y_32,N_global*sizeof(float));
  cudaMemset((void*) x_32,0,N_global*sizeof(float));
  cudaMemset((void*) y_32,0,N_global*sizeof(float));

  // Allocate and convert CSR_LU from double to float
  cudaMalloc((void**)&CSR_LU_32, M*sizeof(float));
#pragma acc parallel loop deviceptr(CSR_LU,CSR_LU_32)
  for (int i=0; i<M; i++){
    CSR_LU_32[i] = (float) CSR_LU[i];
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_mem_alloc_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Setup cusparse
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseCreate(&cusparseHandle_sp);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_cusparse_create_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up M
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseCreateMatDescr(&M_described_sp);
  cusparseSetMatIndexBase(M_described_sp, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatType(M_described_sp, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseStatus = cusparseCreateCsrilu02Info(&M_analyzed_sp);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Info Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_M_setup_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer size for M
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseStatus = cusparseScsrilu02_bufferSize(cusparseHandle_sp, N, M,
                    M_described_sp, CSR_LU_32, CSR_I, CSR_J,
                    M_analyzed_sp, &Mbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Buffer Size Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_M_buffer_size_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cudaMalloc((void**)&Mbuffer, Mbuf_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_M_buffer_alloc_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze M
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseStatus = cusparseScsrilu02_analysis(cusparseHandle_sp,
                    N, M, M_described_sp, CSR_LU_32, CSR_I, CSR_J,
              M_analyzed_sp, M_policy_sp, Mbuffer);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_sp,
                    M_analyzed_sp, &struct_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
      printf(" ERROR! A(%d,%d) is missing\n",
           struct_zero, struct_zero);
      exit(1);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_M_analysis_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set preconditioner (M=LU)
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseStatus = cusparseScsrilu02(cusparseHandle_sp, N, M, M_described_sp,
                    CSR_LU_32, CSR_I, CSR_J, M_analyzed_sp, M_policy_sp, Mbuffer);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Formation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_sp,
                    M_analyzed_sp, &n_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
     printf(" ERROR! M(%d,%d) is zero\n", n_zero, n_zero);
     exit(1);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_M_preconditioner_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Cleanup M buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cudaFree(Mbuffer);
  cusparseDestroyCsrilu02Info(M_analyzed_sp);
  cusparseDestroyMatDescr(M_described_sp);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_M_cleanup_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up L
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  // Create the sparse matrix
  cusparseStatus = cusparseCreateCsr(&L_mat, N, N, M, CSR_I, CSR_J,
      CSR_LU_32, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);
  if (cusparseStatus!=0){
      printf(" ERROR! L CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&L_described_sp);

  // Set filling mode types
  cusparseFillMode_t L_fill_mode = CUSPARSE_FILL_MODE_LOWER;
  cusparseSpMatSetAttribute(L_mat, CUSPARSE_SPMAT_FILL_MODE,
      &L_fill_mode, sizeof(L_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t L_type_diag = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseSpMatSetAttribute(L_mat, CUSPARSE_SPMAT_DIAG_TYPE,
      &L_type_diag, sizeof(L_type_diag));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_L_setup_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up U
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  // Creat the sparse matrix
  cusparseStatus = cusparseCreateCsr(&U_mat, N, N, M, CSR_I, CSR_J,
      CSR_LU_32, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);

  if (cusparseStatus!=0){
      printf(" ERROR! U CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&U_described_sp);

  // Set filling mode types
  cusparseFillMode_t U_fill_mode = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(U_mat, CUSPARSE_SPMAT_FILL_MODE,
      &U_fill_mode, sizeof(U_fill_mode));

  // Set diagonal unit types
  cusparseDiagType_t U_type_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(U_mat, CUSPARSE_SPMAT_DIAG_TYPE,
      &U_type_diag, sizeof(U_type_diag));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_U_setup_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up Dense Vectors
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseCreateDnVec(&DenseVecX, N, x_32, CUDA_R_32F);
  cusparseCreateDnVec(&DenseVecY, N, y_32, CUDA_R_32F);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_dense_vec_setup_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer sizes
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_sp, L_trans_sp,
        &alpha_sp, L_mat, DenseVecX, DenseVecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_sp,&Lbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! L Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_sp, U_trans_sp,
        &alpha_sp, U_mat, DenseVecX, DenseVecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_sp,&Ubuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! U Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_buffer_size_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Allocate buffers
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cudaMalloc((void**)&Lbuffer, Lbuf_size);
  cudaMalloc((void**)&Ubuffer, Ubuf_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_buffer_alloc_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze L
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_sp, L_trans_sp,
        &alpha_sp, L_mat, DenseVecX, DenseVecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_sp, Lbuffer);
    if (cusparseStatus!=0){
      printf(" ERROR! L Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_L_analysis_sp += elapsed_time;

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Analyze U
  //
  ////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(start, 0);
  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_sp, U_trans_sp,
        &alpha_sp, U_mat, DenseVecX, DenseVecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_sp, Ubuffer);
    if (cusparseStatus!=0){
      printf(" ERROR! U Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  time_U_analysis_sp += elapsed_time;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


/* ******************************************************************* */
/* *** Do the Solve Phase: *** */
/* ******************************************************************* */

void lusol_cusparse_sp(double* restrict x)
{
  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Update the Dense Vector (already linked to DenseVecX in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_32)
  for (int i=0;i<N_global;i++){
    x_32[i] = (float) x[i];
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Forward solve (Ly=x) (DenseVecY already linked to y_32 pointer in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_sp, L_trans_sp,
      &alpha_sp, L_mat, DenseVecX, DenseVecY, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, L_described_sp);
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

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_sp, U_trans_sp,
      &alpha_sp, U_mat, DenseVecY, DenseVecX, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, U_described_sp);
  if (cusparseStatus!=0){
      printf(" ERROR! Backward Solve Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Convert result to float precision.
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_32)
  for (int i=0;i<N_global;i++){
    x[i] = (double) x_32[i];
  }

}



/* ******************************************************************* */
/* *** Free the Memory used by cusparse: *** */
/* ******************************************************************* */


void unload_lusol_cusparse_sp()
{
  // Print timing statistics before cleanup
  if (N_global > 0) {
    // Calculate total time by summing all individual timing sections
    float total_time = time_mem_alloc_sp + 
                       time_cusparse_create_sp + 
                       time_M_setup_sp +
                       time_M_buffer_size_sp + 
                       time_M_buffer_alloc_sp + 
                       time_M_analysis_sp +
                       time_M_preconditioner_sp + 
                       time_M_cleanup_sp + 
                       time_L_setup_sp +
                       time_U_setup_sp + 
                       time_dense_vec_setup_sp + 
                       time_buffer_size_sp +
                       time_buffer_alloc_sp + 
                       time_L_analysis_sp + 
                       time_U_analysis_sp;

    printf("\n");
    printf("================================================================================\n");
    printf("load_lusol_cusparse_sp Timing Statistics (N=%d, M=%d)\n", N_global, M_global);
    printf("================================================================================\n");
    printf("  Memory allocation and initialization:     %8.3f ms\n", time_mem_alloc_sp);
    printf("  CUSPARSE handle creation:                  %8.3f ms\n", time_cusparse_create_sp);
    printf("  M - Setup (descriptor, info):              %8.3f ms\n", time_M_setup_sp);
    printf("  M - Buffer size calculation:               %8.3f ms\n", time_M_buffer_size_sp);
    printf("  M - Buffer allocation:                      %8.3f ms\n", time_M_buffer_alloc_sp);
    printf("  M - Analysis:                               %8.3f ms\n", time_M_analysis_sp);
    printf("  M - Preconditioner formation:              %8.3f ms\n", time_M_preconditioner_sp);
    printf("  M - Cleanup:                               %8.3f ms\n", time_M_cleanup_sp);
    printf("  L - Setup (matrix, attributes):             %8.3f ms\n", time_L_setup_sp);
    printf("  U - Setup (matrix, attributes):            %8.3f ms\n", time_U_setup_sp);
    printf("  Dense vectors setup:                       %8.3f ms\n", time_dense_vec_setup_sp);
    printf("  Buffer size calculation (L/U):            %8.3f ms\n", time_buffer_size_sp);
    printf("  Buffer allocation (L/U):                  %8.3f ms\n", time_buffer_alloc_sp);
    printf("  L - Analysis:                              %8.3f ms\n", time_L_analysis_sp);
    printf("  U - Analysis:                              %8.3f ms\n", time_U_analysis_sp);
    printf("--------------------------------------------------------------------------------\n");
    printf("  TOTAL TIME (sum of all sections):          %8.3f ms\n", total_time);
    printf("================================================================================\n");
    printf("\n");
  }

  // Reset timing variables to zero
  time_mem_alloc_sp = 0.0f;
  time_cusparse_create_sp = 0.0f;
  time_M_setup_sp = 0.0f;
  time_M_buffer_size_sp = 0.0f;
  time_M_buffer_alloc_sp = 0.0f;
  time_M_analysis_sp = 0.0f;
  time_M_preconditioner_sp = 0.0f;
  time_M_cleanup_sp = 0.0f;
  time_L_setup_sp = 0.0f;
  time_U_setup_sp = 0.0f;
  time_dense_vec_setup_sp = 0.0f;
  time_buffer_size_sp = 0.0f;
  time_buffer_alloc_sp = 0.0f;
  time_L_analysis_sp = 0.0f;
  time_U_analysis_sp = 0.0f;

  if (L_described_sp != 0) {
    cusparseSpSV_destroyDescr(L_described_sp);
    L_described_sp = 0;
  }
  if (U_described_sp != 0) {
    cusparseSpSV_destroyDescr(U_described_sp);
    U_described_sp = 0;
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
  if (x_32 != NULL) {
    cudaFree(x_32);
    x_32 = NULL;
  }
  if (y_32 != NULL) {
    cudaFree(y_32);
    y_32 = NULL;
  }
  if (CSR_LU_32 != NULL) {
    cudaFree(CSR_LU_32);
    CSR_LU_32 = NULL;
  }
  if (cusparseHandle_sp != NULL) {
    cusparseDestroy(cusparseHandle_sp);
    cusparseHandle_sp = NULL;
  }
  N_global = 0;
  M_global = 0;
  cudaDeviceSynchronize();
}



/* ******************************************************************* */
/* *** Initialize the cusparse functions in SP: *** */
/* ******************************************************************* */

void load_lusol_cusparse_pot2dh_sp(double* restrict CSR_LU, 
          int* restrict CSR_I, int* restrict CSR_J,int N, int M)
{
  N_global_pot2dh = N;

  // Allocate global scratch arrays and initialize to 0.
  cudaMalloc((void**)&x_32_pot2dh,N_global_pot2dh*sizeof(float));
  cudaMalloc((void**)&y_32_pot2dh,N_global_pot2dh*sizeof(float));
  cudaMemset((void*) x_32_pot2dh,0,N_global_pot2dh*sizeof(float));
  cudaMemset((void*) y_32_pot2dh,0,N_global_pot2dh*sizeof(float));

  // Allocate and convert CSR_LU from double to float
  cudaMalloc((void**)&CSR_LU_32_pot2dh, M*sizeof(float));
#pragma acc parallel loop deviceptr(CSR_LU,CSR_LU_32_pot2dh)
  for (int i=0; i<M; i++){
    CSR_LU_32_pot2dh[i] = (float) CSR_LU[i];
  }

  // Setup cusparse.
  cusparseCreate(&cusparseHandle_pot2dh_sp);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateMatDescr(&M_described_pot2dh_sp);
  cusparseSetMatIndexBase(M_described_pot2dh_sp, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatType(M_described_pot2dh_sp, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseStatus = cusparseCreateCsrilu02Info(&M_analyzed_pot2dh_sp);
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

  cusparseStatus = cusparseScsrilu02_bufferSize(cusparseHandle_pot2dh_sp, N, M,
                    M_described_pot2dh_sp, CSR_LU_32_pot2dh, CSR_I, CSR_J,
                    M_analyzed_pot2dh_sp, &Mbuf_size_pot2dh);

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

  cusparseStatus = cusparseScsrilu02_analysis(cusparseHandle_pot2dh_sp,
                    N, M, M_described_pot2dh_sp, CSR_LU_32_pot2dh, CSR_I, CSR_J,
              M_analyzed_pot2dh_sp, M_policy_sp, Mbuffer_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2dh_sp,
                    M_analyzed_pot2dh_sp, &struct_zero);
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

  cusparseStatus = cusparseScsrilu02(cusparseHandle_pot2dh_sp, N, M, M_described_pot2dh_sp,
                    CSR_LU_32_pot2dh, CSR_I, CSR_J, M_analyzed_pot2dh_sp, M_policy_sp, Mbuffer_pot2dh);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Formation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2dh_sp,
                    M_analyzed_pot2dh_sp, &n_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
     printf(" ERROR! M(%d,%d) is zero\n", n_zero, n_zero);
     exit(1);
  }

  cudaFree(Mbuffer_pot2dh);
  cusparseDestroyCsrilu02Info(M_analyzed_pot2dh_sp);
  cusparseDestroyMatDescr(M_described_pot2dh_sp);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Create the sparse matrix
  cusparseStatus = cusparseCreateCsr(&L_mat_pot2dh, N, N, M, CSR_I, CSR_J,
      CSR_LU_32_pot2dh, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);
  if (cusparseStatus!=0){
      printf(" ERROR! L CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&L_described_pot2dh_sp);

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
      CSR_LU_32_pot2dh, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);

  if (cusparseStatus!=0){
      printf(" ERROR! U CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&U_described_pot2dh_sp);

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

  cusparseCreateDnVec(&DenseVecX_pot2dh, N, x_32_pot2dh, CUDA_R_32F);
  cusparseCreateDnVec(&DenseVecY_pot2dh, N, y_32_pot2dh, CUDA_R_32F);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer sizes
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2dh_sp, L_trans_sp,
        &alpha_sp, L_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2dh_sp,&Lbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! L Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2dh_sp, U_trans_sp,
        &alpha_sp, U_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2dh_sp,&Ubuf_size);

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

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2dh_sp, L_trans_sp,
        &alpha_sp, L_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2dh_sp, Lbuffer_pot2dh);
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

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2dh_sp, U_trans_sp,
        &alpha_sp, U_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2dh_sp, Ubuffer_pot2dh);
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

void lusol_cusparse_pot2dh_sp(double* restrict x)
{
  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Update the Dense Vector (already linked to DenseVecX_pot2dh in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_32_pot2dh)
  for (int i=0;i<N_global_pot2dh;i++){
    x_32_pot2dh[i] = (float) x[i];
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Forward solve (Ly=x) (DenseVecY_pot2dh already linked to y_32_pot2dh pointer in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2dh_sp, L_trans_sp,
      &alpha_sp, L_mat_pot2dh, DenseVecX_pot2dh, DenseVecY_pot2dh, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2dh_sp);
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

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2dh_sp, U_trans_sp,
      &alpha_sp, U_mat_pot2dh, DenseVecY_pot2dh, DenseVecX_pot2dh, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2dh_sp);
  if (cusparseStatus!=0){
      printf(" ERROR! Backward Solve Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Convert result to float precision.
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_32_pot2dh)
  for (int i=0;i<N_global_pot2dh;i++){
    x[i] = (double) x_32_pot2dh[i];
  }

}

/* ******************************************************************* */
/* *** Initialize the cusparse functions in SP: *** */
/* ******************************************************************* */

void load_lusol_cusparse_pot2d_sp(double* restrict CSR_LU, int* restrict CSR_I, 
                         int* restrict CSR_J,int N, int M)
{
  N_global_pot2d = N;

  // Allocate global scratch arrays and initialize to 0.
  cudaMalloc((void**)&x_32_pot2d,N_global_pot2d*sizeof(float));
  cudaMalloc((void**)&y_32_pot2d,N_global_pot2d*sizeof(float));
  cudaMemset((void*) x_32_pot2d,0,N_global_pot2d*sizeof(float));
  cudaMemset((void*) y_32_pot2d,0,N_global_pot2d*sizeof(float));

  // Allocate and convert CSR_LU from double to float
  cudaMalloc((void**)&CSR_LU_32_pot2d, M*sizeof(float));
#pragma acc parallel loop deviceptr(CSR_LU,CSR_LU_32_pot2d)
  for (int i=0; i<M; i++){
    CSR_LU_32_pot2d[i] = (float) CSR_LU[i];
  }

  // Setup cusparse.
  cusparseCreate(&cusparseHandle_pot2d_sp);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up M
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseCreateMatDescr(&M_described_pot2d_sp);
  cusparseSetMatIndexBase(M_described_pot2d_sp, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatType(M_described_pot2d_sp, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseStatus = cusparseCreateCsrilu02Info(&M_analyzed_pot2d_sp);
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

  cusparseStatus = cusparseScsrilu02_bufferSize(cusparseHandle_pot2d_sp, N, M,
                    M_described_pot2d_sp, CSR_LU_32_pot2d, CSR_I, CSR_J,
                    M_analyzed_pot2d_sp, &Mbuf_size_pot2d);

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

  cusparseStatus = cusparseScsrilu02_analysis(cusparseHandle_pot2d_sp,
                    N, M, M_described_pot2d_sp, CSR_LU_32_pot2d, CSR_I, CSR_J,
              M_analyzed_pot2d_sp, M_policy_sp, Mbuffer_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Analysis Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2d_sp,
                    M_analyzed_pot2d_sp, &struct_zero);
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

  cusparseStatus = cusparseScsrilu02(cusparseHandle_pot2d_sp, N, M, 
                    M_described_pot2d_sp,CSR_LU_32_pot2d, CSR_I, CSR_J, 
                    M_analyzed_pot2d_sp, M_policy_sp, Mbuffer_pot2d);
  if (cusparseStatus!=0){
      printf(" ERROR! ILU0 Formation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle_pot2d_sp,
                    M_analyzed_pot2d_sp, &n_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
     printf(" ERROR! M(%d,%d) is zero\n", n_zero, n_zero);
     exit(1);
  }

  cudaFree(Mbuffer_pot2d);
  cusparseDestroyCsrilu02Info(M_analyzed_pot2d_sp);
  cusparseDestroyMatDescr(M_described_pot2d_sp);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Set up L
  //
  ////////////////////////////////////////////////////////////////////////////////////

  // Create the sparse matrix
  cusparseStatus = cusparseCreateCsr(&L_mat_pot2d, N, N, M, CSR_I, CSR_J,
      CSR_LU_32_pot2d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);
  if (cusparseStatus!=0){
      printf(" ERROR! L CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&L_described_pot2d_sp);

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
      CSR_LU_32_pot2d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);

  if (cusparseStatus!=0){
      printf(" ERROR! U CSR Matrix Creation Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  // Initialize the data structure
  cusparseSpSV_createDescr(&U_described_pot2d_sp);

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

  cusparseCreateDnVec(&DenseVecX_pot2d, N, x_32_pot2d, CUDA_R_32F);
  cusparseCreateDnVec(&DenseVecY_pot2d, N, y_32_pot2d, CUDA_R_32F);

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Get algorithm buffer sizes
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2d_sp, L_trans_sp,
        &alpha_sp, L_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2d_sp,&Lbuf_size);

  if (cusparseStatus!=0){
      printf(" ERROR! L Buffer Size Error =       %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cusparseStatus = cusparseSpSV_bufferSize(cusparseHandle_pot2d_sp, U_trans_sp,
        &alpha_sp, U_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2d_sp,&Ubuf_size);

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

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2d_sp, L_trans_sp,
        &alpha_sp, L_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2d_sp, Lbuffer_pot2d);
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

  cusparseStatus = cusparseSpSV_analysis(cusparseHandle_pot2d_sp, U_trans_sp,
        &alpha_sp, U_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2d_sp, Ubuffer_pot2d);
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

void lusol_cusparse_pot2d_sp(double* restrict x)
{
  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Update the Dense Vector (already linked to DenseVecX_pot2d in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_32_pot2d)
  for (int i=0;i<N_global_pot2d;i++){
    x_32_pot2d[i] = (float) x[i];
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Forward solve (Ly=x) (DenseVecY_pot2d already linked to y_32_pot2d pointer in load)
  //
  ////////////////////////////////////////////////////////////////////////////////////

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2d_sp, L_trans_sp,
      &alpha_sp, L_mat_pot2d, DenseVecX_pot2d, DenseVecY_pot2d, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, L_described_pot2d_sp);
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

  cusparseStatus = cusparseSpSV_solve(cusparseHandle_pot2d_sp, U_trans_sp,
      &alpha_sp, U_mat_pot2d, DenseVecY_pot2d, DenseVecX_pot2d, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, U_described_pot2d_sp);
  if (cusparseStatus!=0){
      printf(" ERROR! Backward Solve Error =      %s \n",
             cusparseGetErrorString(cusparseStatus));
      exit(1);
  }

  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////////
  //
  // Convert result to float precision.
  //
  ////////////////////////////////////////////////////////////////////////////////////

#pragma acc parallel loop deviceptr(x,x_32_pot2d)
  for (int i=0;i<N_global_pot2d;i++){
    x[i] = (double) x_32_pot2d[i];
  }

}
