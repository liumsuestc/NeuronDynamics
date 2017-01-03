//*****************************//
//     	conductance-based model of E-I networks, Bruno&Wang 2003
//     	Written by Dongping Yang at Hong Kong Baptist University, HK
//		2013-11-01
//****************************************//

//#include "mpi.h"
#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <list>
#include <deque>
#include <queue>
#include <sstream>

using namespace std;

#include "RandNum.h"
CRandNum randNum;

#define RAN 	randNum.GenRandReal_10()
#define RAN00 	randNum.GenRandReal_00()
#define RANDINT randNum.GenRandInt32()
#define Norm	randNum.Gaussian_Noise()

#define NetSize 	5000
#define ExcSize 	4000
#define InhSize		1000

#define Degree		500
#define Degree_X	400
#define Degree_E	400
#define Degree_I	100

#define Delta_t 	0.05    	//unit: ms
#define D			5

#define Error		1e-4
#define Trains		200			//number of spike trains output to file
#define TW			4			//number of different time window
#define	Sample		4			//number of different sample size
#define En			100			//100 ensemble for information capacity

double coupling;

//rand number
inline int rand(int n)
{
	return RANDINT % n;
}


class CNeuron
{
public:
	double x[D];						//x[0] is voltage
										//x[1] is inhibitory synaptic conductance, x[2] is ki,
										//x[3] is external+internal excitatory synaptic conductance, x[4] is ke

	double active_time;					//neuron is in refractory state before this time
	double ex_spike_time;				//external spike time
	deque<double> t_spike;				//time of receiving a spike	
	deque<int>	spike_type;				//external (2), excitatory(1) or inhibitory(0) spike	
	vector<int> linkin;					//input neighbors 
	vector<int> linkout;				//output neighbors

										//insert a spiking time ti into t_spike and spike_type
	int InsertSpike(const double ti, const int si);

	//for individual properties
	int spikes;							//total spikes for firing rate
	bool spike1st;						//before(0) or after(1) first spike
	long double ISIsum, ISIsum2;			//for CVsum of irregularity

											//for patterns
	int state[TW];						//firing or not in each time window of pattern
	int spikesT[TW];					//spikes in each time window of pattern	

										//for outputing spike trains
	bool label;							//selected(1) or not(0) for output the following spike trains
	list<double> SpikeTrain;				//spike train for cross correlations of spiking time
};

int CNeuron::InsertSpike(const double ti, const int si)
{
	int i, min, max, mid;
	if (t_spike.size() == 0) {
		t_spike.push_back(ti);
		spike_type.push_back(si);
	}
	else {
		min = 0;
		max = t_spike.size() - 1;
		while (max - min >= 3) {
			mid = (max + min) >> 1;
			if (ti>t_spike[mid])
				min = mid + 1;
			else
				max = mid - 1;
		}

		for (i = max; i >= min; i--)
			if (ti>t_spike[i]) {
				t_spike.insert(t_spike.begin() + i + 1, ti);
				spike_type.insert(spike_type.begin() + i + 1, si);
				break;
			}
			else if (i == min) {
				t_spike.insert(t_spike.begin() + i, ti);
				spike_type.insert(spike_type.begin() + i, si);
				break;
			}
	}

	return 0;
}

class CNetwork
{
public:
	// for Runge-Kutta 2! algorithm
	double dx[D], k1[D], k2[D], xtemp[D];
	inline	double Voltagefunc(double xx[]);
	inline	double gifunc(double xx[]);
	inline	double kifunc(double xx[]);
	inline	double gefunc(double xx[]);
	inline	double kefunc(double xx[]);

	double (CNetwork::*function[D])(double xx[]);
	int RK2(int ith, double timestep, int start, int end);

	vector<CNeuron> neuron;
	double Vrest, Vreset, Vth, Vleak;

	double fex, ge, gi;					//external input rate, synaptic strength for E and I	
	double gee, gie, gei, gii;				//e-->e,e-->i,i-->e,i-->i	
	double scale, balance;

	double tau_e, tau_i;
	double tau_rp_e, tau_rp_i;
	double tau_rise, tau_delay;
	double tau_decay_e, tau_decay_i;
	int commu_index[NetSize];


	//auxillary variables
	double Gext, Ge, Gi;
	double tau, tau_rp;

	//parameter
	double startTime;					//starting time, from here to 0 is transient
	double TotalTime;					//total running time	
	int TotalSpikesE;					//total excitatory spikes

	int Initial();  					//initiate state
	int Set();     						//set parameters	

	int CreateRandomNet();  			//random network
	int CreateRanHomNet();				//random homogeneous network with homogeneous input connections
	int CreateComNet(double InterPro1, double InterPro2);  			//random community network
	int evolution(char file[]);			//dynamics
};

int CNetwork::Set()
{
	CNetwork::function[0] = &CNetwork::Voltagefunc;
	CNetwork::function[1] = &CNetwork::gifunc;
	CNetwork::function[2] = &CNetwork::kifunc;
	CNetwork::function[3] = &CNetwork::gefunc;
	CNetwork::function[4] = &CNetwork::kefunc;

	//unit: mV		
	Vrest = -70;			Vreset = -60;
	Vth = -50;			Vleak = -70;

	//unit: ms	
	tau_e = 20;			tau_i = 10;
	tau_rp_e = 2;			tau_rp_i = 1;
	tau_delay = 1;		tau_rise = 0.5;

	//synaptic strength, unit: nS*R   R=1e8	
	scale = 0.08;			balance = 0.5;
	ge = 1.25*balance*scale;				gi = scale;		//external AMPA 
	gee = balance*scale;					gie = scale;		//AMPA
	gei = 1.25*balance*scale * 12;			gii = scale * 12;		//GABA	

	fex = 2.5*Degree_X*1e-3;								//external input, unit: kHz
	return 0;
}

int CNetwork::Initial()
{
	int i, j, tw;

	for (i = 0; i<NetSize; i++) {
		neuron[i].x[0] = Vrest + (Vth - Vrest)*RAN;
		for (j = 1; j<D; j++)
			neuron[i].x[j] = 0;

		neuron[i].active_time = startTime;
		neuron[i].ex_spike_time = startTime - log(RAN00) / fex;
		

		if (i>=0&& i < ExcSize / 2)
			commu_index[i] = 1;
		if (i>= ExcSize / 2&&i<ExcSize)
			commu_index[i] = 2;
		if (i>=ExcSize&& i < ExcSize+InhSize/2)
			commu_index[i] = 1;
		if(i>= ExcSize + InhSize / 2&&i<NetSize)
			commu_index[i] = 2;
		neuron[i].spikes = 0;
		neuron[i].spike1st = 0;
		neuron[i].ISIsum = 0;
		neuron[i].ISIsum2 = 0;

		for (tw = 0; tw<TW; tw++) {
			neuron[i].state[tw] = 0;
			neuron[i].spikesT[tw] = 0;
		}

		neuron[i].label = 0;
	}
	return 0;
}

//create a random network
//input :NetSize
int CNetwork::CreateRandomNet()
{
	int i, j;
	double linkProb = Degree / (double)NetSize;
	neuron.resize(NetSize);
	for (i = 0; i < NetSize; i++) {
		for (j = 0; j < NetSize; j++) {
			if ((RAN <= linkProb) && (i != j)) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}

	return 0;
}

int CNetwork::CreateComNet(double InterPro1, double InterPro2)
{
	int i, j;
	double linkProb = 0.2;
	neuron.resize(NetSize);
	// 1-2000
	for (i = 0; i < 2000; i++) {
		for (j = 0; j < 2000; j++) {
			if ((RAN <= linkProb) && (i != j)&&find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}
	//4000-4500
	for (i = 4000; i < 4500; i++) {
		for (j = 4000; j < 4500; j++) {
			if ((RAN <= linkProb) && (i != j) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}
	// 2000-4000
	for (i = ExcSize /2; i < ExcSize; i++) {
		for (j = ExcSize /2; j < ExcSize; j++) {
			if ((RAN <= linkProb) && (i != j) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}
	//4500-5000
	for (i = ExcSize + InhSize / 2; i < ExcSize + InhSize; i++) {
		for (j = ExcSize + InhSize / 2; j < ExcSize + InhSize; j++) {
			if ((RAN <= linkProb) && (i != j) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}
	// 1-2000 4001-4500
	for (i = 0; i < 2000; i++) {
		for (j = 4000; j < 4500; j++) {
			if ((RAN <= linkProb) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
				//continue;
			}

			if ((RAN <= linkProb) && find(neuron[j].linkout.begin(), neuron[j].linkout.end(), i) == neuron[j].linkout.end()) {
				//j-->i
				neuron[j].linkout.push_back(i);
				neuron[i].linkin.push_back(j);
			}
		}
	}
	// 2000-4000 4501-5000
	for (i = ExcSize / 2; i < ExcSize; i++) {
		for (j = ExcSize + InhSize / 2; j < ExcSize + InhSize; j++) {
			if ((RAN <= linkProb) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
				//continue;
			}

			if ((RAN <= linkProb) && find(neuron[j].linkout.begin(), neuron[j].linkout.end(), i) == neuron[j].linkout.end()) {
				//j-->i
				neuron[j].linkout.push_back(i);
				neuron[i].linkin.push_back(j);
			}
		}
	}
	//communityA->B
	for (i = 0; i < ExcSize/2; i++) {
		for (j = ExcSize /2; j < ExcSize; j++) {
			if ((RAN < InterPro1) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}

	//communityB->A
	for (i = ExcSize /2; i < ExcSize; i++) {
		for (j = 0; j < ExcSize /2; j++) {
			if ((RAN < InterPro2) && find(neuron[i].linkout.begin(), neuron[i].linkout.end(), j) == neuron[i].linkout.end()) {
				//i-->j
				neuron[i].linkout.push_back(j);
				neuron[j].linkin.push_back(i);
			}
		}
	}

	return 0;
}

int CNetwork::CreateRanHomNet()
{
	int i, j, index, select;
	neuron.resize(NetSize);
	vector<int> source;
	for (i = 0; i < NetSize; i++) {
		//receive excitatory input connections	
		source.clear();
		for (j = 0; j<ExcSize; j++)
			if (j != i)
				source.push_back(j);
		while (neuron[i].linkin.size()<Degree_E) {
			index = rand(source.size());
			select = source[index];

			neuron[i].linkin.push_back(select);
			neuron[select].linkout.push_back(i);

			source.erase(source.begin() + index);
		}

		//receive inhibitory input connections
		source.clear();
		for (j = ExcSize; j<NetSize; j++)
			if (j != i)
				source.push_back(j);
		while (neuron[i].linkin.size()<Degree) {
			index = rand(source.size());
			select = source[index];

			neuron[i].linkin.push_back(select);
			neuron[select].linkout.push_back(i);

			source.erase(source.begin() + index);
		}
	}
	return 0;
}


inline double CNetwork::Voltagefunc(double xx[])
{
	//return (Vleak-xx[0]-10*xx[1]+60*xx[3])/tau;
	return (Vleak*(1 + xx[1]) - xx[0] * (1 + xx[1] + xx[3])) / tau;
}
inline	double CNetwork::gifunc(double xx[])
{
	return (xx[2] - xx[1]) / tau_decay_i;
}
inline	double CNetwork::kifunc(double xx[])
{
	return -xx[2] / tau_rise;
}
inline	double CNetwork::gefunc(double xx[])
{
	return (xx[4] - xx[3]) / tau_decay_e;
}
inline	double CNetwork::kefunc(double xx[])
{
	return -xx[4] / tau_rise;
}

int CNetwork::RK2(int ith, double timestep, int start, int end)
{
	int j;

	for (j = start; j<end; j++)
		k1[j] = (this->*function[j])(neuron[ith].x);
	for (j = start; j<end; j++)
		xtemp[j] = neuron[ith].x[j] + timestep*k1[j];
	for (j = start; j<end; j++)
		k2[j] = (this->*function[j])(xtemp);

	for (j = start; j<end; j++)
		dx[j] = timestep*(k1[j] + k2[j]) / 2;
	return 0;
}

int CNetwork::evolution(char file[])
{
	int i, j, tw, n, step = 20;
	double begin, end;
	double time_next, time = startTime;
	double temp, interval, t_spike, CV;
	int flag2, flag = 0;

	int temp1 = 0, count = 0;
	int temp2 = 0;
	ofstream avalanche,spikeseres;
	char file_ava[40];
	sprintf_s(file_ava, "Avalanche_%s", file);

	CreateComNet(0.025, 0);
	//CreateRandomNet();
	Initial();
	
	///////////////////////////////////

	//pattern neuron&spike size distributions
	char file_kprob[40];
	ofstream fout_kprob;
	sprintf_s(file_kprob, "pat_dis_%s", file);
	fout_kprob.open(file_kprob);
	fout_kprob << "%1st line: TimeWindow\tSampleSize" << endl;
	fout_kprob << "%2nd&3rd lines: neuron size distribution" << endl;
	fout_kprob << "%4th&5th lines: spike size distribution" << endl;

	//selected spike trains
	char file_SpikeTrain[40];
	ofstream fout_SpikeTrain;
	sprintf_s(file_SpikeTrain, "SpikeTrains_%s", file);
	fout_SpikeTrain.open(file_SpikeTrain);
	fout_SpikeTrain.precision(9);
	fout_SpikeTrain << "Neuron#\tSpikingTime" << endl;

	// for entropy based on 100 randomly selected ensemble: 
	// size=10, only e neurons, time window T=20ms;	
	double T0[TW] = { 2,5,10,20 };
	double T[TW] = { 2,5,10,20 };
	double p_event[TW];
	for (tw = 0; tw<TW; tw++)
		p_event[tw] = T[tw] / TotalTime;
	int sample_size[Sample];

	//for statistics of patterns	
	int Spikes_En[En][Sample][TW], SpikesSum[Sample][TW];
	double P0[En][Sample][TW], entropy[En][Sample][TW];
	double PatRatioSum[Sample][TW], EntropySum[Sample][TW], EfficiencySum[Sample][TW];
	for (n = 0; n<Sample; n++) {
		sample_size[n] = 10 + n * 10;

		for (tw = 0; tw<TW; tw++) {
			SpikesSum[n][tw] = 0;
			PatRatioSum[n][tw] = 0;
			EntropySum[n][tw] = 0;
			EfficiencySum[n][tw] = 0;

			for (j = 0; j<En; j++) {
				Spikes_En[j][n][tw] = 0;
				P0[j][n][tw] = 0;
				entropy[j][n][tw] = 0;
			}
		}
	}

	//pattern distritions			
	string pat;	 ostringstream convert;
	vector<string> Pat[En][Sample][TW];
	vector<double> Pat_dis[En][Sample][TW];

	int pat_n, pat_s;
	vector<int> Pat_n[Sample][TW], Pat_s[Sample][TW];
	vector<double> Pat_s_dis[Sample][TW], Pat_n_dis[Sample][TW];

	int index, select;
	vector<int> source;		vector<int>	labeled;
	vector<int> ensemble[En][Sample];

	//select sample_size neurons as an ensemble base
	for (j = 0; j<En; j++) {
		for (n = 0; n<Sample; n++) {
			source.resize(ExcSize);
			for (i = 0; i<ExcSize; i++)
				source[i] = i;
			for (i = 0; i<sample_size[n]; i++) {
				index = rand(source.size());
				select = source[index];
				ensemble[j][n].push_back(select);

				if (n == Sample - 1)
					if (labeled.size()<Trains)
						if (neuron[select].label == 0) {
							neuron[select].label = 1;
							labeled.push_back(select);
						}

				source.erase(source.begin() + index);
			}
		}
	}


	////////////////////////////////// 		

	while (time<TotalTime) {
		cout << time << endl;
		time_next = time + Delta_t;
		// (1) update each neuron
		for (i = 0; i<NetSize; i++) {
			// 0), set neuron type
			if (i<ExcSize) {
				tau = tau_e;				tau_rp = tau_rp_e;
				Gext = ge*tau / tau_rise;	Ge = gee*tau / tau_rise;		Gi = gei*tau / tau_rise;
			}
			else {
				tau = tau_i;				tau_rp = tau_rp_i;
				Gext = gi*tau / tau_rise;	Ge = gie*tau / tau_rise;		Gi = gii*tau / tau_rise;
			}

			// 1), external input of spikes, record spiking time
			while (neuron[i].ex_spike_time <= time_next) {
				neuron[i].InsertSpike(neuron[i].ex_spike_time, 3);
				interval = -log(RAN00) / fex;
				neuron[i].ex_spike_time += interval;
			}

			// 2), integrate all incoming spikes
			end = time;
			while (end<time_next) {
				//1. set begin and end
				begin = end;
				if ((neuron[i].t_spike.size() == 0) || (neuron[i].t_spike.front()>time_next))
					end = time_next;
				else
					end = neuron[i].t_spike.front();

				//2. integrate the interval [begin,end]
				if (end - begin>Error) {
					//case 1: begin<end<=active_time, totally in refractory period
					if (end <= neuron[i].active_time) {
						interval = end - begin;
						RK2(i, interval, 1, D);
						for (j = 1; j<D; j++)
							neuron[i].x[j] += dx[j];
					}
					//case 2: begin<active_time<end, partially in refractory period
					else if (begin<neuron[i].active_time) {
						interval = neuron[i].active_time - begin;
						RK2(i, interval, 1, D);
						for (j = 1; j<D; j++)
							neuron[i].x[j] += dx[j];

						begin = neuron[i].active_time;
						interval = end - begin;
						RK2(i, interval, 0, D);
						for (j = 1; j<D; j++)
							neuron[i].x[j] += dx[j];

						if (dx[0]<Vth - neuron[i].x[0])
							neuron[i].x[0] += dx[0];
						else
							flag = 1;
					}
					//case 3: active_time<=begin<end, totally active
					else {
						interval = end - begin;
						RK2(i, interval, 0, D);
						for (j = 1; j<D; j++)
							neuron[i].x[j] += dx[j];

						if (dx[0]<Vth - neuron[i].x[0])
							neuron[i].x[0] += dx[0];
						else
							flag = 1;
					}

					//checking neuron firing or not
					if (flag) {
						flag = 0;
						//a. get spiking time						
						t_spike = begin + interval*(Vth - neuron[i].x[0]) / dx[0];

						if (time_next>0) {
							//b1. for each neuron's rate and CV
							neuron[i].spikes++;
							if (i >= 0 &&i < ExcSize/2) //record num of spike neuron at time t
							{
								temp1++;
							}
							if (i >= ExcSize / 2 &&i < ExcSize) //record num of spike neuron at time t
							{
								temp2++;
							}
							if (neuron[i].spike1st == 0)
								neuron[i].spike1st = 1;
							else {
								interval = t_spike + tau_rp - neuron[i].active_time;
								neuron[i].ISIsum += interval;
								neuron[i].ISIsum2 += interval*interval;
							}

							//b2. for neuron's state in patterns
							for (tw = 0; tw<TW; tw++) {
								neuron[i].state[tw] = 1;
								neuron[i].spikesT[tw]++;
							}

							//b3. for outputing neuron spike trains
							if (neuron[i].label == 1)
								neuron[i].SpikeTrain.push_back(t_spike);
						}

						//c. spreading spikes to neighbors i->j		
						for (j = 0; j < neuron[i].linkout.size(); j++)
						{
							neuron[neuron[i].linkout[j]].InsertSpike(t_spike + tau_delay, int(i < ExcSize) + abs(commu_index[i] - commu_index[neuron[i].linkout[j]]));
						}
						//d. reset neuron potential and active_time	
						neuron[i].x[0] = Vreset;
						neuron[i].active_time = t_spike + tau_rp;
					}
				}

				//3. a spike comes in: 0-Inh spike, 1-inner Exc, 2-Outer Exc, 3-Background input
				if (end<time_next) {
					switch (neuron[i].spike_type.front())
					{
					case 0:
					{
						neuron[i].x[2] += Gi;break; 
					}
					case 1:
					{
						neuron[i].x[4] += Ge; break;
					}
					case 2:
					{
						neuron[i].x[4] += coupling; break;
					}
					case 3:
					{
						neuron[i].x[4] += Gext; break;
					}
					default:
						break;
					} 
					neuron[i].t_spike.pop_front();
					neuron[i].spike_type.pop_front();
				}
			}
		}
		/*if (int(time / Delta_t) % 1 == 0)  //avalanche is defined as nonempty spike sequenece
		{
			if (temp2 != 0)
			{
				count += temp2;
			}
			else
			{
				if (count != 0)
				{
					avalanche.open(file_ava, ios::app);
					avalanche << count << "\t";
					avalanche.close();
					count = 0;
				}
			}
			temp2 = 0;
		}*/

		/////////// update time				
		time = time_next;

		if (time_next>0) {
			spikeseres.open("spikeseries1.txt", ios::app);
			spikeseres << temp1 << "\t";
			spikeseres.close();
			temp1 = 0;

			spikeseres.open("spikeseries2.txt", ios::app);
			spikeseres << temp2 << "\t";
			spikeseres.close();
			temp2 = 0;

			// do statistics of each pattern
			for (tw = 0; tw<TW; tw++) {
				T[tw] -= Delta_t;
				if (T[tw] <= 0) {
					T[tw] = T0[tw];
					for (j = 0; j<En; j++) {
						for (n = 0; n<Sample; n++) {
							convert.str("");
							pat_s = 0;	pat_n = 0;
							for (i = 0; i<sample_size[n]; i++) {
								select = ensemble[j][n][i];
								// get pattern string
								if (neuron[select].spikesT[tw]<10)
									convert << neuron[select].spikesT[tw];
								else if (neuron[select].spikesT[tw] == 10)
									convert << "A";
								pat_s += neuron[select].spikesT[tw];
								pat_n += neuron[select].state[tw];
								Spikes_En[j][n][tw] += neuron[select].spikesT[tw];
							}
							pat = convert.str();

							if (pat_s == 0)
								P0[j][n][tw] += p_event[tw];
							else {
								//pattern distribution

								flag2 = 0;
								for (i = 0; i<Pat[j][n][tw].size(); i++)
									if (pat.compare(Pat[j][n][tw][i]) == 0) {
										Pat_dis[j][n][tw][i] += p_event[tw];
										flag2 = 1;	break;
									}
								if (flag2 == 0) {
									Pat[j][n][tw].push_back(pat);
									Pat_dis[j][n][tw].push_back(p_event[tw]);
								}
								//spike size distribution

								flag2 = 0;
								for (i = 0; i<Pat_s[n][tw].size(); i++)
									if (pat_s == Pat_s[n][tw][i]) {
										Pat_s_dis[n][tw][i] += p_event[tw];
										flag2 = 1;	break;
									}
								if (flag2 == 0) {
									Pat_s[n][tw].push_back(pat_s);
									Pat_s_dis[n][tw].push_back(p_event[tw]);
								}

								//active neuruon size distribution

								flag2 = 0;
								for (i = 0; i < Pat_n[n][tw].size(); i++)
									if (pat_n == Pat_n[n][tw][i]) {
										Pat_n_dis[n][tw][i] += p_event[tw];
										flag2 = 1;	break;
									}
								if (flag2 == 0) {
									Pat_n[n][tw].push_back(pat_n);
									Pat_n_dis[n][tw].push_back(p_event[tw]);
								}
							}
						}
					}
					//reset neuron's state
					for (i = 0; i<NetSize; i++) {
						neuron[i].state[tw] = 0;
						neuron[i].spikesT[tw] = 0;
					}
				}
			}
		}
	}



	//2. output the distribution of pattern spike size and neuron size
	for (tw = 0; tw<TW; tw++) {
		for (n = 0; n<Sample; n++) {
			fout_kprob << T0[tw] << "\t" << sample_size[n] << endl;

			for (i = 0; i<Pat_n[n][tw].size(); i++)
				fout_kprob << Pat_n[n][tw][i] << "\t";
			fout_kprob << endl;
			for (i = 0; i<Pat_n_dis[n][tw].size(); i++)
				fout_kprob << Pat_n_dis[n][tw][i] / double(En) << "\t";
			fout_kprob << endl;

			for (i = 0; i<Pat_s[n][tw].size(); i++)
				fout_kprob << Pat_s[n][tw][i] << "\t";
			fout_kprob << endl;
			for (i = 0; i<Pat_s_dis[n][tw].size(); i++)
				fout_kprob << Pat_s_dis[n][tw][i] / double(En) << "\t";
			fout_kprob << endl;
		}
	}


	//4. output selected spike trains
	for (i = 0; i<Trains; i++) {
		select = labeled[i];
		fout_SpikeTrain << select;
		while (!neuron[select].SpikeTrain.empty()) {
			fout_SpikeTrain << "\t" << neuron[select].SpikeTrain.front();
			neuron[select].SpikeTrain.pop_front();
		}
		fout_SpikeTrain << endl;
	}

	//fout_Ind.close();
	fout_kprob.close();
	//fout_pat.close();
	fout_SpikeTrain.close();

	return 0;
}


int main()
{
	//////////////////////////////////////////////////////////////////////////////
	// for MPI
//	int rank, size, num;
	/*MPI::Init(argc, argv);
	string process_name;
	char name[MPI_MAX_PROCESSOR_NAME];
	size = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();
	//////////////////////////////////////////////////////////////////////////	 
	ifstream fin("para");
	fin >> num;
	fin.close();
	num = 161 + rank + num * 40;*/
	cout << "input coupling"<<endl;
	cin >> coupling;


	CNetwork network;
	network.startTime = -1e3;
	network.TotalTime = 1e4;

	network.balance = 0.5;
	//network.tau_decay_e = 2;
	cout << "input tde" << endl;
	cin >> network.tau_decay_e;
	//network.tau_decay_i = 14;
	cout << "input tdi" << endl;
	cin >> network.tau_decay_i;
	network.Set();

	char filename[100];
	sprintf_s(filename, "tde=%.2lf_tdi=%.1lf.txt", network.tau_decay_e, network.tau_decay_i);

	network.evolution(filename);
	network.neuron.clear();
	
	//cout << abs(2-1)<<endl;
	//system("pause");
	//MPI::Finalize();
	
	return 0;
}

