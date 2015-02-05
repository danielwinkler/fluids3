/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com

  Attribute-ZLib license (* See additional part 4)

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  4. Any published work based on this code must include public acknowledgement
     of the origin. This includes following when applicable:
	   - Journal/Paper publications. Credited by reference to work in text & citation.
	   - Public presentations. Credited in at least one slide.
	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
	 Retaining this additional license term is required in derivative works.
	 Acknowledgement may be provided as:
	   Publication version:  
	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
	 	  Fluid Simulator. Published online at: http://fluids3.com
	   Single line (slides or app credits):
	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)

 Notes on Clause 4:
  The intent of this clause is public attribution for this contribution, not code use restriction. 
  Both commerical and open source projects may redistribute and reuse without code release.
  However, clause #1 of ZLib indicates that "you must not claim that you wrote the original software". 
  Clause #4 makes this more specific by requiring public acknowledgement to be extended to 
  derivative licenses. 

*/

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>

#ifndef CUDA_KERNEL
    // Prefix Sum
//    #include "prefix_sum.cu"


// from prefix sum

// Define this to more rigorously avoid bank conflicts,
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS

//TODO
    #define NUM_BANKS       16
    #define LOG_NUM_BANKS    4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS)
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
//
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// excellent paper "Prefix sums and their applications".
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//

template <bool isNP2> __device__ void loadSharedChunkFromMem (float *s_data, const float *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB )
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);         // compute spacing to avoid bank conflicts
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    s_data[ai + bankOffsetA] = g_idata[mem_ai];     // Cache the computational window in shared memory pad values beyond n with zeros

    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi];
    }
}


template <bool isNP2> __device__ void loadSharedChunkFromMemInt (int *s_data, const int *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB )
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);         // compute spacing to avoid bank conflicts
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    s_data[ai + bankOffsetA] = g_idata[mem_ai];     // Cache the computational window in shared memory pad values beyond n with zeros

    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi];
    }
}

template <bool isNP2> __device__ void storeSharedChunkToMem(float* g_odata, const float* s_data, int n, int ai, int bi, int mem_ai, int mem_bi,int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    g_odata[mem_ai] = s_data[ai + bankOffsetA];         // write results to global memory
    if (isNP2) { // compile-time decision
        if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB];
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB];
    }
}
template <bool isNP2> __device__ void storeSharedChunkToMemInt (int* g_odata, const int* s_data, int n, int ai, int bi, int mem_ai, int mem_bi,int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    g_odata[mem_ai] = s_data[ai + bankOffsetA];         // write results to global memory
    if (isNP2) { // compile-time decision
        if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB];
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB];
    }
}


template <bool storeSum> __device__ void clearLastElement( float* s_data, float *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;      // zero the last element in the scan so it will propagate back to the front
    }
}

template <bool storeSum> __device__ void clearLastElementInt ( int* s_data, int *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;      // zero the last element in the scan so it will propagate back to the front
    }
}


__device__ unsigned int buildSum(float *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;

    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
    }
    return stride;
}
__device__ unsigned int buildSumInt (int *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;

    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
    }
    return stride;
}

__device__ void scanRootToLeaves(float *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

__device__ void scanRootToLeavesInt (int *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum> __device__ void prescanBlock(float *data, int blockIndex, float *blockSums)
{
    int stride = buildSum (data);               // build the sum in place up the tree
    clearLastElement<storeSum> (data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves (data, stride);            // traverse down tree to build the scan
}
template <bool storeSum> __device__ void prescanBlockInt (int *data, int blockIndex, int *blockSums)
{
    int stride = buildSumInt (data);               // build the sum in place up the tree
    clearLastElementInt <storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeavesInt (data, stride);            // traverse down tree to build the scan
}

__global__ void uniformAdd (float *g_data, float *uniforms, int n, int blockOffset, int baseIndex)
{
    __shared__ float uni;
    if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

    __syncthreads();
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}
__global__ void uniformAddInt (int *g_data, int *uniforms, int n, int blockOffset, int baseIndex)
{
    __shared__ int uni;
    if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

    __syncthreads();
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

#endif

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;

	// Particle & Grid Buffers
	struct bufList {
		float3*			mpos;
		float3*			mvel;
		float3*			mveleval;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;		
		uint*			mgcell;
		uint*			mgndx;
		uint*			mclr;			// 4 byte color

		uint*			mcluster;

		char*			msortbuf;

		uint*			mgrid;	
		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;
	};

	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_VEL			(sizeof(float3))
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))

	// Fluid Parameters (stored on both host and device)
	struct FluidParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum;
		int				chk;
		float			pdist, pmass, prest_dens;
		float			pextstiff, pintstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity;
		float			AL, AL2, VL, VL2;

		float			d2, rd2, vterm;		// used in force calculation		 
		
		float			poly6kern, spikykern, lapkern;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;

		int				gridAdj[64];
	};

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS		16
	#define LOG_NUM_BANKS	 4


	#ifndef CUDA_KERNEL

		// Declare kernel functions that are available to the host.
		// These are defined in kern.cu, but declared here so host.cu can call them.

		__global__ void insertParticles ( bufList buf, int pnum );
		__global__ void countingSortIndex ( bufList buf, int pnum );		
		__global__ void countingSortFull ( bufList buf, int pnum );		
		__global__ void computeQuery ( bufList buf, int pnum );	
		__global__ void computePressure ( bufList buf, int pnum );		
		__global__ void computeForce ( bufList buf, int pnum );
		__global__ void computePressureGroup ( bufList buf, int pnum );
		__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts );

		__global__ void countActiveCells ( bufList buf, int pnum );		

		void updateSimParams ( FluidParams* cpufp );

		// NOTE: Template functions must be defined in the header
		template <bool storeSum, bool isNP2> __global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ float s_data[];
			loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
			prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);
			storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
		}
		template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ int s_dataInt [];
			loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
			prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums);
			storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
		}
		__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);	
		__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);	
	#endif
	

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

	
#endif
