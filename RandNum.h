/******************************************************************************
This program is adapted based on trimmed version of MT19937.
To get the original version,
contact <http://www.math.keio.ac.jp/~matumoto/emt.html

RandNum.h -

$Author: Hai Lin $
$Date: 2010/09/27 $

Copyright (C) 1993-2003 Yukihiro Matsumoto

/********************************************************************************/
/********************************************************************************
A C-program for MT19937, with initialization improved 2002/1/26.
Coded by Takuji Nishimura and Makoto Matsumoto.

Before using, initialize the state by using init_genrand(seed)
or init_by_array(init_key, key_length).

Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. The names of its contributors may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Any feedback is very welcome.
http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
********************************************************************************/

#ifndef RANDNUM_H_HAILIN
#define RANDNUM_H_HAILIN

#include <math.h>
#include <time.h>


double PI = 2.0*asin(1.0);


//高精度随机数产生器
class CRandNum
{
private:
	int N;
	int M;
	unsigned long MATRIX_A;						//constant vector a
	unsigned long UMASK;						//most significant w-r bits
	unsigned long LMASK;						//least significant r bits

	unsigned long *state; 						//the array for the state vector

	int initleft;
	int initf;
	unsigned long *next;

public:
	CRandNum();
	~CRandNum();

	void RandSeed();
	void InitGenRand(unsigned long s);			//initializes state[N] with a seed
	void InitByArray(unsigned long init_key[],	//initialize by an array with array-length
		unsigned long key_length);
	void NextState(void);
	unsigned long GenRandInt32(void); 			//random number on [0,0xffffffff]-interval
	unsigned long GenRandInt31(void); 			//random number on [0,0x7fffffff]-interval
	double GenRandReal_11(void);				//random number on [0,1]-real-interval
	double GenRandReal_10(void);				//random number on [0,1)-real-interval
	double GenRandReal_00(void);				//random number on (0,1)-real-interval
	double GenRandRes53_10(void); 				//random number on [0,1) with 53-bit resolution
	double Gaussian_Noise(void);


};

CRandNum::CRandNum()
{
	N = 624;
	M = 397;
	MATRIX_A = 0x9908b0dfUL;					//constant vector a */
	UMASK = 0x80000000UL;						//most significant w-r bits */
	LMASK = 0x7fffffffUL;						//least significant r bits */

	state = new unsigned long[N];

	int initleft = 1;
	int initf = 0;

	RandSeed();
}

CRandNum::~CRandNum()
{
	delete[] state;
}

void CRandNum::RandSeed()
{
	//srand(1859);
	srand((unsigned)time( NULL ));
	unsigned long init[4] = { unsigned long(rand()), unsigned long(rand()), unsigned long(rand()), unsigned long(rand()) };
	unsigned long length = 4;
	InitByArray(init, length);
}

// initializes state[N] with a seed
void CRandNum::InitGenRand(unsigned long s)
{
	int j;
	state[0] = s & 0xffffffffUL;
	for (j = 1; j < N; j++)
	{
		state[j] = (1812433253UL * (state[j - 1] ^ (state[j - 1] >> 30)) + j);

		// See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
		// In the previous versions, MSBs of the seed affect
		// only MSBs of the array state[].
		// 2002/01/09 modified by Makoto Matsumoto

		state[j] &= 0xffffffffUL;  // for >32 bit machines
	}
	initleft = 1;
	initf = 1;
}

// initialize by an array with array-length
// init_key is the array for initializing keys
// key_length is its length
void CRandNum::InitByArray(unsigned long initKey[], unsigned long keyLength)
{
	int i, j, k;
	InitGenRand(19650218UL);
	i = 1;
	j = 0;
	k = (N > int(keyLength)) ? (N) : int(keyLength);
	for (; k; k--)
	{
		// non linear
		state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525UL))
			+ initKey[j] + j;

		// for WORDSIZE > 32 machines
		state[i] &= 0xffffffffUL;

		i++;
		j++;
		if (i >= N)
		{
			state[0] = state[N - 1]; i = 1;
		}

		if (j >= keyLength)
		{
			j = 0;
		}
	}
	for (k = N - 1; k; k--)
	{
		// non linear
		state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941UL))
			- i;

		// for WORDSIZE > 32 machines
		state[i] &= 0xffffffffUL;

		i++;
		if (i >= N)
		{
			state[0] = state[N - 1];
			i = 1;
		}
	}

	// MSB is 1; assuring non-zero initial array
	state[0] = 0x80000000UL;
	initleft = 1;
	initf = 1;
}

void CRandNum::NextState(void)
{
	unsigned long *p = state;
	int j;

	// if init_genrand() has not been called,
	// a default initial seed is used
	if (initf == 0)
	{
		InitGenRand(5489UL);
	}

	initleft = N;
	next = state;

	for (j = N - M + 1; --j; p++)
	{
		*p = p[M] ^ (((((p[0]) & UMASK) | ((p[1]) & LMASK)) >> 1)
			^ ((p[1]) & 1UL ? MATRIX_A : 0UL));
	}

	for (j = M; --j; p++)
	{
		*p = p[M - N] ^ (((((p[0]) & UMASK) | ((p[1]) & LMASK)) >> 1)
			^ ((p[1]) & 1UL ? MATRIX_A : 0UL));
	}

	*p = p[M - N] ^ (((((p[0]) & UMASK) | ((state[0]) & LMASK)) >> 1)
		^ ((state[0]) & 1UL ? MATRIX_A : 0UL));
}

// generates a random number on [0,0xffffffff]-interval
unsigned long CRandNum::GenRandInt32(void)
{
	unsigned long y;

	if (--initleft == 0)
	{
		NextState();
	}

	y = *next++;

	// Tempering
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

// generates a random number on [0,0x7fffffff]-interval
unsigned long CRandNum::GenRandInt31(void)
{
	unsigned long y;

	if (--initleft == 0)
	{
		NextState();
	}
	y = *next++;

	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return (long)(y >> 1);
}

// generates a random number on [0,1]-real-interval
double CRandNum::GenRandReal_11(void)
{
	unsigned long y;

	if (--initleft == 0)
	{
		NextState();
	}

	y = *next++;

	// Tempering
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	// divided by 2^32-1
	return (double)y * (1.0 / 4294967295.0);
}

// generates a random number on [0,1)-real-interval
double CRandNum::GenRandReal_10(void)
{
	unsigned long y;

	if (--initleft == 0)
	{
		NextState();
	}
	y = *next++;

	// Tempering
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	// divided by 2^32
	return (double)y * (1.0 / 4294967296.0);
}

// generates a random number on (0,1)-real-interval
double CRandNum::GenRandReal_00(void)
{
	unsigned long y;

	if (--initleft == 0)
	{
		NextState();
	}
	y = *next++;

	// Tempering
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	// divided by 2^32
	return ((double)y + 0.5) * (1.0 / 4294967296.0);
}

// generates a random number on [0,1) with 53-bit resolution
double CRandNum::GenRandRes53_10(void)
{
	unsigned long a = GenRandInt32() >> 5;
	unsigned long b = GenRandInt32() >> 6;
	return(a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}


double CRandNum::Gaussian_Noise(void)
{
	double pi = 2.0*asin(1.0);
	double gauss_noise = sqrt(-4.0 * log(GenRandReal_11())) * sin(2.0 * pi * GenRandReal_10());

	if (gauss_noise > 5)
		gauss_noise = 5;
	else if (gauss_noise < -5)
		gauss_noise = -5;

	return gauss_noise;
}

#endif
