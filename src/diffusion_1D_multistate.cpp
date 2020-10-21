// 
// Program for estimating 1D diffusion model with multiple energy profiles from trajectories 
// by K. Okazaki, June 2020
//
// 'nr3.h', 'hmm.h', 'eigen_sym.h', 'dynpro.h' are from Numerical Recipes, Third Edition   
//
// Compile:
// $ g++ -o ../diffusion_1D_multistate -std=c++11 -O2 diffusion_1D_multistate.cpp
// For debugging, add '-ggdb' option.
//
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath> // for exp(), log(), sqrt()       
#include "./nr3.h"
//#include "./nr3mod.h" // g++ 4.8
#include "./hmm.h" // for HMM
#include "./hmm_pd.h" // for pd-HMM
#include "./eigen_sym.h" // for Symmeig
#include "./dynpro.h" // for DP
#include <algorithm> // for "min/max_element"
#include <random>

// global variables
const double pi = std::acos(-1);
const int ns = 2; // number of states (energy profiles)
const int nbin = 27; // number of bins
const int ksym = nbin*nbin;
////// parameters that can be read from "para" file
int ctrl = 3;
// ctrl=0 is for optimizing F and D by HMM; ctrl=1 is for optimizing switching path by DP, ctrl=2 is for optimizing switching rates by pd-HMM
// ctrl=3 is for optimizing F and D (and switching rates) by pd-HMM
int ctrl_sa = 0; // 1 for MAP estimate by simulated annealing
int replicate = 1; // replicate=1 means multiple energy surfaces and diff coefs are replicated and the energy surface is shifted
int nmc = 1000;
int nmc_update_stop=500;
int nmc_sa = 2000;
double k_sa = 0.02;
double dx = 0.1;
double xlim_low = -0.35; // lower bound of position x 
double dt0 = 0.000001;
int lagstep = 400;
double dt = lagstep * dt0;
double prior_dv = 6.0; //6.0;
double prior_sigma2 = 0.1*0.1; //0.1*0.1;
int mcrate = 1; // mcrate=1 means Monte Calro search in ratef_switch and rateb_switch (ctrl=0,3), mcrate=2 only optimizes ratef_switch
double xswitch[] = {0.8, 1.2}; // switching region used in ctrl=1,2
double ratef_switch = 1.0/0.010; // forward rate for state switching:: this is for HMM
double rateb_switch = 0.0; // backward rate for state switching:: this is for HMM
double xshift = 1.0; // for replicate=1, 2nd energy surface is shifted by xshift after replication
//////
double countmat[nbin][nbin];
double rate[nbin][nbin];
double transprob[ns][ksym];
MatDoub trans_mat(nbin,nbin); // nr3 style
std::vector< std::vector<double> >  trajorg, trajstp;
std::vector< std::vector<double> >  trajseg;
std::vector<double>  xdp;


// functions
void readtraj(char *c);
void readtraj0(char *c);
void savetraj(const std::vector<double> &v);
void copy_trajorg_to_trajseg();
void readpara();
void read_initialmodel(double (&v1)[ns][nbin],double (&w1)[ns]);
void transition_count(const std::vector< std::vector<double> > &v2, int lagstep, double dx);
void transition_count2();
void init_rate_matrix(int nbin, double v1[], double w1);
void transition_matrix(int nbin, double v1[], double w1);
double log_likelihood(int nbin, double v1[], double w1);
int mc_optimize(double v1[], double w1);
double prob_switch_2state(int sb, int sa, double ratef, double rateb);
double prob_switch(int sb, int sa, double ratef, double rateb);
void trajseg_to_trajsymbol(const std::vector<double> &v, int lagstep, double dx,std::vector<int> &v3);
void savehmmprob(const MatDoub &v, const std::string f);
double log_prior(double (&v1)[ns][nbin]);
double log_likelihood_hmm(double (&v1)[ns][nbin], double (&w1)[ns]);
int mc_optimize_hmm(double (&v1)[ns][nbin], double (&w1)[ns]);
int path_optimize(double (&v1)[ns][nbin], double (&w1)[ns]);
Doub costfunc(Int jj, Int kk, Int ii);
void savepathopt(const VecInt &v, const std::string f);
int mc_optimize_hmm_rates(double (&v1)[ns][nbin], double (&w1)[ns]);
void switching_region(const std::vector<double> &v, VecInt &sr);


int main(int argc, char *argv[])
{
  // Command line
  if (argc==1){
    std::cout <<"usage: " <<argv[0] <<" [traj file1] [traj file2] ..." <<endl;
    return 0;
  }

  // Read original and step trajectories from files
  int ntraj=argc-1;
  std::cout <<"# of trajectories: "<<ntraj <<endl;
  for (int i=1;i<=ntraj;i++) {
    char *infile;  
    infile = argv[i];
    //readtraj(infile);    
    readtraj0(infile);    
  }
  // Check
  // savetraj(trajorg[1]);
  // return 0;

  // Processing trajectory data
  // Use trajectories as they are
  copy_trajorg_to_trajseg();

  // Read parameters from "para" file
  readpara();


  double v[ns][nbin];
  double w[ns];

  if (ctrl==0 || ctrl==3) {

    // Bayesian inference
    // 1) Initialize diffusion model
    for (int i=0; i<ns; i++) {
      for (int j=0;j<nbin;j++) {
		v[i][j] = 0.0;
      }
      //w[i] = 1.0;
      //w[i] = 7.5;
      w[i] = 9.0;
    }
    // to make it start from a condition that satisfies prior
    if (replicate==1 && ns>=2) {
      int addbin = (int) (xshift/dx);
	  // peripheral region on right side
	  for (int ii=0;ii<ns-1;ii++) {
		for (int i=nbin-(ns-1-ii)*addbin;i<nbin;i++) {
		  double dvtemp = prior_dv*(i-(nbin-(ns-1-ii)*addbin)+1);
		  v[ii][i] = v[ii][nbin-(ns-1-ii)*addbin-1] + dvtemp;
		}
	  }
	  // peripheral region on left side
	  for (int ii=1;ii<ns;ii++) {
		for (int i=0;i<=ii*addbin-1;i++) {
		  double dvtemp = prior_dv*((ii*addbin)-i);
		  v[ii][i] = v[ii][ii*addbin] + dvtemp;
		}
	  }
    }

    // 2) Sampling posterior distribution of the model
    if (ns==1) {
      // Count transitions between bins
      transition_count2();
      // Optimize v and w
      mc_optimize(v[0],w[0]);
    } else if (ns>=2) {
      // Optimize v and w by HMM
      mc_optimize_hmm(v,w);
    }

  } else if (ctrl==1) {

    // optimization of hidden-state path by DP
    // read the model from 'initialmodel.out'
    read_initialmodel(v,w);
    // Optimization of hidden-state pathways
    path_optimize(v,w);

  } else if (ctrl==2) {

    // optimization of switching rates by pd-HMM
    // read the model from 'initialmodel.out'
    read_initialmodel(v,w);
    // Optimization of switching rates
    mc_optimize_hmm_rates(v,w);

  } 

  return 0;

}


void readtraj(char *c)
{
  std::vector<double>   trajorg1, trajstp1;
  // Read one line at a time into the variable line:
  std::ifstream fi(c);
  std::string line;
  while (std::getline(fi,line)) {
    std::stringstream  lineStream(line);
    double value;
    int icl=0;
    // Read an number at a time from the line
    while(lineStream >> value) {
      icl++;
      // Add the numbers from a line to a 1D array (vector)
      if (icl==4) {
	trajorg1.push_back(value);
      }
      if (icl==6) {
	trajstp1.push_back(value);
      }
    }
  }
  std::cout <<"trajorg1 size:" <<trajorg1.size() <<endl; 
  std::cout <<"trajstp1 size:" <<trajstp1.size() <<endl; 
  // When all the numbers have been read, add the 1D array
  // into a 2D array (as one line in the 2D array)
  trajorg.push_back(trajorg1);
  trajstp.push_back(trajstp1);
}

void readtraj0(char *c)
{
  std::vector<double>   trajorg1;
  // Read one line at a time into the variable line:
  std::ifstream fi(c);
  std::string line;
  while (std::getline(fi,line)) {
    std::stringstream  lineStream(line);
    double value;
    int icl=0;
    // Read an number at a time from the line
    while(lineStream >> value) {
      icl++;
      // Add the numbers from a line to a 1D array (vector)
      if (icl==2) {
		trajorg1.push_back(value);
      }
    }
  }
  std::cout <<"trajorg1 size:" <<trajorg1.size() <<endl; 
  // When all the numbers have been read, add the 1D array
  // into a 2D array (as one line in the 2D array)
  trajorg.push_back(trajorg1);
}


void savetraj(const std::vector<double> &v)
{
  std::ofstream fo("test.out");
  for (int i=0; i<v.size(); i++) {
    fo <<v[i] <<endl;
  }
  fo.close();
}


void copy_trajorg_to_trajseg()
{
  int ntraj;
  ntraj = trajorg.size();
  std::cout <<"# of trajectories: "<<ntraj <<endl;
  // loop over trajectories
  for (int itraj=0; itraj<ntraj; itraj++) {
    trajseg.push_back(trajorg[itraj]);
  }
}


void readpara()
{
  std::cout <<"readpara()" <<endl;
  // Read one line at a time into the variable line:
  std::ifstream fi("para");
  std::string line;
  while (std::getline(fi,line)) {
    std::stringstream  lineStream(line);
	std::string word;
	std::string varname;
    int icl=0;
    // Read an number at a time from the line
    while(lineStream >> word) {
      icl++;	  
      if (icl==1) {
		varname = word;
		std::cout <<varname <<" ";
      }
	  if (icl==2) {
		if (varname=="ctrl") {
		  ctrl = std::stoi(word);
		  std::cout <<ctrl <<endl;
		}
		if (varname=="ctrl_sa") {
		  ctrl_sa = std::stoi(word);
		  std::cout <<ctrl_sa <<endl;
		}
		if (varname=="replicate") {
		  replicate = std::stoi(word);
		  std::cout <<replicate <<endl;
		}
		if (varname=="nmc") {
		  nmc = std::stoi(word);
		  std::cout <<nmc <<endl;
		}
		if (varname=="nmc_update_stop") {
		  nmc_update_stop = std::stoi(word);
		  std::cout <<nmc_update_stop <<endl;
		}
		if (varname=="nmc_sa") {
		  nmc_sa = std::stoi(word);
		  std::cout <<nmc_sa <<endl;
		}
		if (varname=="k_sa") {
		  k_sa = std::stod(word);
		  std::cout <<k_sa <<endl;
		}
		if (varname=="dx") {
		  dx = std::stod(word);
		  std::cout <<dx <<endl;
		}
		if (varname=="xlim_low") {
		  xlim_low = std::stod(word);
		  std::cout <<xlim_low <<endl;
		}
		if (varname=="dt0") {
		  dt0 = std::stod(word);
		  std::cout <<dt0 <<endl;
		}
		if (varname=="lagstep") {
		  lagstep = std::stoi(word);
		  std::cout <<lagstep <<endl;
		}
		if (varname=="dt") {
		  dt = std::stod(word);
		  std::cout <<dt <<endl;
		}
		if (varname=="prior_dv") {
		  prior_dv = std::stod(word);
		  std::cout <<prior_dv <<endl;
		}
		if (varname=="prior_sigma2") {
		  prior_sigma2 = std::stod(word);
		  std::cout <<prior_sigma2 <<endl;
		}
		if (varname=="mcrate") {
		  mcrate = std::stoi(word);
		  std::cout <<mcrate <<endl;
		}
		if (varname=="xswitch") {
		  xswitch[0] = std::stod(word);
		  std::cout <<xswitch[0] <<" ";
		}
		if (varname=="ratef_switch") {
		  ratef_switch = std::stod(word);
		  std::cout <<ratef_switch <<endl;
		}
		if (varname=="rateb_switch") {
		  rateb_switch = std::stod(word);
		  std::cout <<rateb_switch <<endl;
		}
		if (varname=="xshift") {
		  xshift = std::stod(word);
		  std::cout <<xshift <<endl;
		}
	  } else if (icl==3) {
		if (varname=="xswitch") {
		  xswitch[1] = std::stod(word);
		  std::cout <<xswitch[1] <<endl;
		}
	  }

    }
  }

}


void read_initialmodel(double (&v1)[ns][nbin],double (&w1)[ns])
{
  // initialmodel.out should have 
  // nbin columns for v1 in first line
  // one column for w1 in second line
  // this should be repeated for multiple states
  std::ifstream fi("initialmodel.out");
  // v1
  // Read one line at a time
  std::string line;
  int irow=0;
  while(std::getline(fi,line)){
    std::stringstream  lineStream(line);
    double value;
    for (int i=0; i<ns; i++) {
      if (irow==2*i) {
	int icl=0;
	// Read an number at a time from the line
	while(lineStream >> value) {
	  v1[i][icl] = value;
	  icl++;
	}
      } else if (irow==2*i+1) {
	lineStream >> value;
	w1[i] = log(value/(dx*dx));
      } 
    }
    irow += 1;
  }
}


void transition_count(const std::vector< std::vector<double> > &trajin, int lagstep, double dx)
{
  // initialize count matrix
  for (int i=0;i<nbin;i++) {
    for (int j=0;j<nbin;j++) {
      countmat[i][j] = 0.0;
    }
  }
  std::ofstream fo("trajin_discrete.out");
  // loop over traj. segments
  for (int i=0;i<trajin.size();i++) {
    int nt = trajin[i].size();
    // assign bin number (discretization)
    int binnum[nt];
    for (int j=0;j<nt;j++) {
      binnum[j] = (int)((trajin[i][j]-xlim_low)/dx);
      //if (binnum[j]<0 || binnum[j]>=nbin) {
      //std::cout <<"!! warning: binnum=" <<binnum[j] <<endl;
      //}
      // printout
      double x = dx*binnum[j] + 0.5*dx + xlim_low;
      fo <<trajin[i][j] <<" " <<x <<endl;
    }
    fo <<endl;
    // count transitions
    for (int j=0;j<nt-lagstep;j++) {
      if (binnum[j]<0 || binnum[j]>=nbin)
		continue;
      if (binnum[j+lagstep]<0 || binnum[j+lagstep]>=nbin)
		continue;
      countmat[binnum[j+lagstep]][binnum[j]] += 1.0;
    }
  }
  fo.close();  

  // save count matrix
  std::ofstream fo2("count_matrix.out");
  for (int i=0;i<nbin;i++) {
    for (int j=0;j<nbin;j++) {
      fo2 <<countmat[i][j] <<" ";
    }
    fo2 <<endl;
  }
  fo2.close();
}

void transition_count2()
{
  // initialize count matrix
  for (int i=0;i<nbin;i++) {
    for (int j=0;j<nbin;j++) {
      countmat[i][j] = 0.0;
    }
  }
  // loop over traj. segments
  for (int i=0;i<trajseg.size();i++) {
    int nt = trajseg[i].size();
	int nt2 = nt/lagstep + 1;
    for (int it2=0;it2<nt2-1;it2++) {
	  int it = it2*lagstep;
	  int binnum1, binnum2;
      binnum1 = (int)((trajseg[i][it]-xlim_low)/dx);
      binnum2 = (int)((trajseg[i][it+lagstep]-xlim_low)/dx);
	  if (binnum1<0) {
		binnum1 = 0;
	  } else if (binnum1>=nbin) {
		binnum1 = nbin-1;
	  }
	  if (binnum2<0) {
		binnum2 = 0;
	  } else if (binnum2>=nbin) {
		binnum2 = nbin-1;
	  }
      countmat[binnum2][binnum1] += 1.0;
    }
  }

}

void init_rate_matrix(int nbin, double v1[], double w1)
{
  // initialize rate matrix from potential vector v and diffusion vector w = log(D(i)/delta^2)
  // diffusion coefficient D(i+1/2)
  double d;
  d = exp(w1);
  // sqrt(probability p(i))
  double p2[nbin];
  for (int i=0; i<nbin; i++) {
    p2[i] = exp(-0.5*v1[i]);
  }
  // rate matrix
  for (int i=0; i<nbin; i++) {
    for (int j=0; j<nbin; j++) {
      rate[i][j] = 0.0;
    }
  }
  for (int i=0; i<nbin-1; i++) {
    rate[i+1][i] = d*p2[i+1]/p2[i]; // d[i]*exp(-0.5*(V[i+1]-V[i]))
    rate[i][i+1] = d*p2[i]/p2[i+1]; // d[i]*exp(-0.5*(V[i]-V[i+1]))
  }
  rate[0][0] = -rate[1][0];
  rate[nbin-1][nbin-1] = -rate[nbin-2][nbin-1];
  for (int i=1; i<nbin-1; i++) {
    rate[i][i] = -rate[i-1][i] - rate[i+1][i];
  }
}

void transition_matrix(int nbin, double v1[], double w1)
{
  init_rate_matrix(nbin, v1, w1);

  // Calculating matrix exponential exp(K*dt)
  MatDoub rate_sym(nbin,nbin); // symmetric rate matrix
  for (int i=0; i<nbin; i++) {
    for (int j=0; j<nbin; j++) {
      if (j==i) {
	rate_sym[i][j] = rate[i][j];
	continue;
      }
      rate_sym[i][j] = sqrt(rate[i][j]*rate[j][i]);
    }
  }
  // call Symmeig
  Symmeig myeig(rate_sym);
  MatDoub eigvec(nbin,nbin),eigveci(nbin,nbin),eigvalexp(nbin,nbin);
  // exp of eigenvalues, matrix of eigenvectors of K
  for (int i=0; i<nbin; i++) {
    for (int j=0; j<nbin; j++) {
      if (j==i)
	eigvalexp[i][j] = exp(myeig.d[i]*dt);
      else
	eigvalexp[i][j] = 0.0;
      eigvec[i][j] = myeig.z[i][0]*myeig.z[i][j];
      eigveci[i][j] = myeig.z[j][i]/myeig.z[j][0];
    }
  }  
  // transition matrix T = V*exp(lamda*dt)*V-1
  for (int i=0; i<nbin; i++) {
    for (int j=0; j<nbin; j++) {
      double eleij = 0.0;
      for (int k=0; k<nbin; k++) {
	eleij += eigvec[i][k]*(eigveci[k][j]*eigvalexp[k][k]);
      }
      trans_mat[i][j] = eleij;
    }
  }
  //check
  //std::cout <<"sum of each column in transition matrix" <<endl;
  //for (int i=0; i<nbin; i++) {
  //  double s = 0.0;
  //  for (int j=0; j<nbin; j++) {
  //    s += trans_mat[j][i];
  //  }
  //  std::cout <<s <<endl;
  //}  
}

double log_likelihood(int nbin, double v1[], double w1)
{
  transition_matrix(nbin,v1,w1); // calculate "trans_mat"
  // log likelihood
  double tiny = 1.0e-20;
  double loglike = 0.0;
  for (int i=0;i<nbin;i++) {
    for (int j=0;j<nbin;j++) {
      double p = trans_mat[i][j];
	  if (p<tiny){
		//std::cout <<p <<endl;
	  	p = tiny;
	  }
      loglike += log(p)*countmat[i][j];
    }
  }
  return loglike;
}

int mc_optimize(double v1[], double w1)
{
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0, 1);

  // basic parameters
  double temp=1.0;
  //  int nmc=5000; // # of MC moves
  int num_mc_update=50; // # of moves between width adjustments
  // int nmc_update_stop=1000; // after nmc_update_stop, widths are no longer changed
  double dv[nbin]; // width of MC moves
  for (int i=0;i<nbin;i++) {
    dv[i] = 0.025;
  }
  double dw=0.025; // width of MC moves

  double naccv=0.0; // # accepted v moves
  double naccw=0.0; // # accepted w moves
  double naccv_update[nbin];
  for (int i=0;i<nbin;i++) {
    naccv_update[i] = 0.0;
  }
  double naccw_update=0.0;

  // initial log-likelihood
  double log_like,log_like_try;
  log_like = log_likelihood(nbin,v1,w1);
  std::cout <<"initial log-likelihood: " <<log_like <<endl;

  // open output files
  std::ofstream fo1("pmf.out");
  std::ofstream fo2("diffcoef.out");
  std::ofstream fo3("loglike.out");

  // simulated annealing?
  int nmc2 = nmc;
  if (ctrl_sa==1) {
	nmc2 += nmc_sa;
  }

  // loop over MC moves
  for (int imc=0;imc<nmc2;imc++) {

	// for simulated annealing
	if (ctrl_sa==1 && imc>=nmc) {
	  temp = exp(-k_sa*(imc-nmc));
	}

    // loop over bins to modify V
    for (int i=0;i<nbin;i++) {
      double v1mod[nbin];
      for (int j=0;j<nbin;j++) { // coping v1 to v1mod
		v1mod[j] = v1[j];
      }
      v1mod[i] += dv[i]*(dis(gen)-0.5); // Each call to dis(gen) generates a new random double
      log_like_try = log_likelihood(nbin,v1mod,w1);
      double dlog=log_like_try - log_like;
      double r=dis(gen); // Each call to dis(gen) generates a new random double
      if (r<exp(dlog/temp)) { // accept move
		v1[i] = v1mod[i];
		naccv += 1.0;
		naccv_update[i] += 1.0;
		log_like = log_like_try;
      }
    }
    // Modify W
    double w1mod;
    w1mod = w1;
    w1mod += dw*(dis(gen)-0.5); // Each call to dis(gen) generates a new random double
    log_like_try = log_likelihood(nbin,v1,w1mod);
    double dlog=log_like_try - log_like;
    double r=dis(gen); // Each call to dis(gen) generates a new random double
    if (r<exp(dlog/temp)) { // accept move
      w1 = w1mod;
      naccw += 1.0;
      naccw_update += 1.0;
      log_like = log_like_try;
    }
    // Print intermediate results to screen
    if (imc%10==0 || imc==nmc-1) {
      double ntry1=(imc+1.0)*nbin;
      double ntry2=imc+1.0;
      std::cout <<imc <<" " <<log_like <<" " <<naccv/ntry1 <<" " <<naccw/ntry2 <<endl;
      std::cout <<imc <<" ";
      for (int i=0;i<nbin;i++) {
		std::cout <<v1[i] <<" ";
      }
      std::cout <<endl;
      std::cout <<imc <<" " <<exp(w1) <<endl;
    }
    // Print intermediate results to files
    if (imc%10==0 || imc==nmc-1) {
      double ntry1=(imc+1.0)*nbin;
      double ntry2=imc+1.0;
      fo3 <<imc <<" " <<temp <<" " <<log_like <<" " <<naccv/ntry1 <<" " <<naccw/ntry2 <<endl;
      for (int i=0;i<nbin;i++) {
		fo1 <<v1[i] <<" ";
      }
      fo1 <<endl;
      fo2 <<w1 <<endl;
    }
    // Update MC move widths
    if (imc>=nmc_update_stop) // stop adjusting widths
      num_mc_update = 0;
    if (num_mc_update>0) {
      if ((imc+1)%num_mc_update == 0) {
	// adjust widths to target 0.3 acceptance ratio
		for (int i=0;i<nbin;i++) {
		  dv[i] = dv[i]*exp(0.5*(naccv_update[i]/num_mc_update - 0.3));
		  naccv_update[i] = 0.0;
		}
		dw = dw*exp(0.5*(naccw_update/num_mc_update - 0.3));
		naccw_update = 0.0;
      }
    }
    
  }
  // eigen values of transition matrix
  //  std::ofstream fo4("eigenvalues.out");
  // symmetric transition matrix
  //MatDoub trans_mat_sym(nbin,nbin);
  //for (int i=0; i<nbin; i++) {
  //  for (int j=0; j<nbin; j++) {
  //    trans_mat_sym[i][j] = sqrt(trans_mat[i][j]*trans_mat[j][i]);
  //  }
  //}
  // call Symmeig
  //Symmeig myeig2(trans_mat_sym);
  //for (int i=0;i<nbin;i++) {
  //  fo4 <<myeig2.d[i] <<endl;
  //}

  //close output files
  fo1.close();
  fo2.close();
  fo3.close();
  //fo4.close();

  return 0;
}


double prob_switch_2state(int sb, int sa, double ratef, double rateb) 
{
  //std::cout <<"...prob_switch_2state... " <<endl;
  double prob=0.0;
  if (sb==0) {
    if (sa==sb) {
      //prob = 1.0 - ratef*dt;
      //prob = exp(-ratef*dt);
	  prob = (rateb + ratef*exp(-(ratef+rateb)*dt))/(ratef+rateb);
    } else if (sa==sb+1) {
      //prob = ratef*dt;
      //prob = 1.0 - exp(-ratef*dt);
	  prob = (ratef - ratef*exp(-(ratef+rateb)*dt))/(ratef+rateb);
    }
  } else if (sb==1) {
    if (sa==sb) {
      //prob = 1.0 - rateb*dt;
      //prob = exp(-rateb*dt);
	  prob = (ratef + rateb*exp(-(ratef+rateb)*dt))/(ratef+rateb);	  
    } else if (sa==sb-1) {
      //prob = rateb*dt;
      //prob = 1.0 - exp(-rateb*dt);
	  prob = (rateb - rateb*exp(-(ratef+rateb)*dt))/(ratef+rateb);
    }
  } 
  return prob;
}

double prob_switch(int sb, int sa, double ratef, double rateb) 
{
  //std::cout <<"...prob_switch... " <<endl;
  double prob=0.0;
  if (sb==0) {
    if (sa==sb) {
      prob = 1.0 - ratef*dt;
    } else if (sa==sb+1) {
      prob = ratef*dt;
    }
  } else if (sb==ns-1) {
    if (sa==sb) {
      prob = 1.0 - rateb*dt;
    } else if (sa==sb-1) {
      prob = rateb*dt;
    }
  } else {
    if (sa==sb) {
      prob = 1.0 - ratef*dt - rateb*dt;
    } else if (sa==sb-1) {
      prob = rateb*dt;
    } else if (sa==sb+1) {
      prob = ratef*dt;
    }
  }
  return prob;
}


void trajseg_to_trajsymbol(const std::vector<double> &trajin, int lagstep, double dx, std::vector<int> &trajsymbol) 
{
  //std::cout <<"...trajseg_to_trajsymbol..." <<endl;
  int nt = trajin.size();
  int nt2 = nt/lagstep + 1;
  //std::cout <<"nt=" <<nt <<"nt2-1=" <<nt2-1 <<endl;
  for (int it2=0;it2<nt2-1;it2++) {
    int it = it2*lagstep;
    int binnum1, binnum2;
    binnum1 = (int)((trajin[it]-xlim_low)/dx);
    binnum2 = (int)((trajin[it+lagstep]-xlim_low)/dx);
    if (binnum1<0) {
      binnum1 = 0;
    } else if (binnum1>=nbin) {
      binnum1 = nbin-1;
    }
    if (binnum2<0) {
      binnum2 = 0;
    } else if (binnum2>=nbin) {
      binnum2 = nbin-1;
    }
    int isymbol = binnum1 * nbin + binnum2;
    trajsymbol.push_back(isymbol);
  }
}

void savehmmprob(const MatDoub &v, const std::string f)
{
  //std::cout <<"nrow:" <<v.nrows() <<endl;
  //std::cout <<"ncol:" <<v.ncols() <<endl;
  std::ofstream fo(f);
  for (int i=0; i<v.nrows(); i++) {
    for (int j=0;j<v.ncols();j++) {
      fo <<v[i][j] <<" ";
    }
    fo <<endl;
  }
  fo.close();
}

double log_prior(double (&v)[ns][nbin])
{
  double log_pp=0.0;
  if (replicate==1) {
	// prior for replicating energy surfaces
	int addbin = (int) (xshift/dx);
	// For state ii,
	// right-side: ibin=nbin-(ns-ii-1)*addbin ~ nbin-1 should monotoneously increase 
	// left-side:  ibin=0 ~ ii*addbin-1 should monotoneously decrease 
	//
	// right side
	for (int ii=0;ii<ns-1;ii++) {
	  for (int i=nbin-(ns-1-ii)*addbin;i<nbin;i++) {
	    double dvtemp = prior_dv*(i-(nbin-(ns-1-ii)*addbin)+1);
	    log_pp += -(v[ii][i]-(v[ii][nbin-(ns-1-ii)*addbin-1]+dvtemp))*(v[ii][i]-(v[ii][nbin-(ns-1-ii)*addbin-1]+dvtemp))/(2.0*prior_sigma2);
	  }
	}
	//for (int i=nbin-addbin;i<nbin;i++) {
	//  double dvtemp = prior_dv*(i-(nbin-addbin)+1);
	//  log_pp += -(v[0][i]-(v[0][nbin-addbin-1]+dvtemp))*(v[0][i]-(v[0][nbin-addbin-1]+dvtemp))/(2.0*prior_sigma2);
	//}
	
	// left side
	for (int ii=1;ii<ns;ii++) {
	  for (int i=0;i<=ii*addbin-1;i++) {
	    double dvtemp = prior_dv*((ii*addbin)-i);
	    log_pp += -(v[ii][i]-(v[ii][ii*addbin]+dvtemp))*(v[ii][i]-(v[ii][ii*addbin]+dvtemp))/(2.0*prior_sigma2);
	  }
	}
	//for (int i=0;i<=addbin-1;i++) {
	//  double dvtemp = prior_dv*(addbin-i);
	//  log_pp += -(v[1][i]-(v[1][addbin]+dvtemp))*(v[1][i]-(v[1][addbin]+dvtemp))/(2.0*prior_sigma2);
	//}
  }
  return log_pp;
}

double log_likelihood_hmm(double (&v)[ns][nbin], double (&w)[ns])
{
  //std::cout <<"...log_likelihood_hmm... " <<endl;
  MatDoub p_switch(ns,ns); // 'a' in HMM
  for (int i=0;i<ns;i++) {
    for (int j=0;j<ns;j++) {
	  if (ns==2) {
		p_switch[i][j] = prob_switch_2state(i,j,ratef_switch,rateb_switch);
	  } else if (ns>2) {
		p_switch[i][j] = prob_switch(i,j,ratef_switch,rateb_switch);
	  }
      //std::cout <<p_switch[i][j] <<" ";
    }
    //std::cout <<endl;
  }
  int ksym = nbin*nbin;
  double q = 1.0/(double)nbin;
  MatDoub diffuse(ns,ksym); // 'b' in HMM
  for (int i=0;i<ns;i++) {
    transition_matrix(nbin, v[i], w[i]);
    //for (int ib=0;ib<nbin;ib++) {
    //  q += exp(-v[i][ib]);
    //}
    for (int j=0;j<ksym;j++) {
      int ibin = j/nbin;
      int jbin = j%nbin;
      diffuse[i][j] = trans_mat[jbin][ibin] * q; // transition prob. ibin -> jbin
    }
  }
  // Analysis over trajectories
  int ntraj = trajseg.size();
  double loglike_hmm = 0.0;
  for (int itraj=0;itraj<ntraj;itraj++) {
    //std::cout <<"itraj= " <<itraj <<endl;
    // Observed transitions converted to integer symbols
    std::vector<int> trajsymbol;
    trajseg_to_trajsymbol(trajseg[itraj],lagstep,dx,trajsymbol);
    int nt = trajsymbol.size();
    VecInt obsdata(nt); // 'obs' in HMM
    for (int it=0;it<nt;it++) {
      obsdata[it] = trajsymbol[it];
    }
    if (ctrl==0) { // normal HMM
      // Call HMM
      //std::cout <<"...call myhmm..." <<endl;
      HMM myhmm(p_switch,diffuse,obsdata);
      myhmm.forwardbackward();
      //std::cout <<"...forwardbackward done." <<endl;
      loglike_hmm += myhmm.loglikelihood(); 
	  //std::cout <<"loglike_hmm = " <<loglike_hmm <<endl;
      std::string filename = "hmm_prob_traj" + std::to_string(itraj) + ".out";
      savehmmprob(myhmm.pstate,filename);

    } else if (ctrl==2 || ctrl==3) { // pd-HMM
      // 'sr' switching region or not
      VecInt sr(nt); // 'sr' in HMM
      switching_region(trajseg[itraj],sr);
      // call HMM
      HMMpd myhmmpd(p_switch,diffuse,obsdata,sr);
      myhmmpd.forwardbackward2();
      //std::cout <<"...forwardbackward done." <<endl;
      loglike_hmm += myhmmpd.loglikelihood(); //std::cout <<"loglike_hmm = " <<loglike_hmm <<endl;
      std::string filename = "hmmpd_prob_traj" + std::to_string(itraj) + ".out";
      savehmmprob(myhmmpd.pstate,filename);      
    }
	// correction of normalization factor q
 	loglike_hmm += - nt*log(q);
  }
  return loglike_hmm;
}


int mc_optimize_hmm(double (&v)[ns][nbin], double (&w)[ns])
{
  //std::cout <<"...mc_optimize_hmm... " <<endl;
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0, 1);

  // basic parameters
  double temp=1.0;
  //int nmc=2000; // # of MC moves
  int num_mc_update=50; // # of moves between width adjustments
  //int nmc_update_stop=1000; // after nmc_update_stop, widths are no longer changed
  double dv[ns][nbin]; // width of MC moves
  double dw[ns]; // width of MC moves
  for (int j=0;j<ns;j++) {
    for (int i=0;i<nbin;i++) {
      dv[j][i] = 0.025;
    }
    dw[j]=0.025; 
  }

  double naccv[ns]; // # accepted v moves
  double naccw[ns]; // # accepted w moves
  double naccv_update[ns][nbin];
  double naccw_update[ns];
  for (int j=0;j<ns;j++) {
    naccv[j] = 0.0;
    naccw[j] = 0.0;
    for (int i=0;i<nbin;i++) {
      naccv_update[j][i] = 0.0;
    }
    naccw_update[j] = 0.0;
  }

  // MC of rate: mcrate==1
  double drate[2];
  double naccrate_update[2];
  for (int j=0;j<2;j++) {
    drate[j] = 2.0;
    naccrate_update[j] = 0.0;
  }


  // initial log-likelihood
  double log_like,log_like0,log_like1;
  double log_like_try,log_like_try0,log_like_try1;
  log_like0 = log_likelihood_hmm(v,w);
  log_like1 = log_prior(v);
  log_like =  log_like0 + log_like1;
  std::cout <<"initial log-likelihood: " <<log_like <<" " <<log_like0 <<" " <<log_like1  <<endl;

  // open output files
  std::ofstream fo1("pmf.out");
  std::ofstream fo2("diffcoef.out");
  std::ofstream fo3("loglike.out");
  std::ofstream fo5("rate.out");
  std::ofstream fo6("mcwidth.out");

  // simulated annealing?
  int nmc2 = nmc;
  if (ctrl_sa==1) {
	nmc2 += nmc_sa;
  }

  // loop over MC moves
  for (int imc=0;imc<nmc2;imc++) {

	// for simulated annealing
	if (ctrl_sa==1 && imc>=nmc) {
	  temp = exp(-k_sa*(imc-nmc));
	}

    if (replicate==1) { // replicate energy surface with a shift

      int addbin = (int) (xshift/dx);

      for (int ii=0;ii<ns;ii++) {

		// copying v to vmod
		double vmod[ns][nbin];
		for (int jj=0;jj<ns;jj++) { 
		  for (int j=0;j<nbin;j++) { 
			vmod[jj][j] = v[jj][j];
		  }
		}

		// Modify V: loop over bins
		for (int i=0;i<nbin;i++) {
		  vmod[ii][i] += dv[ii][i]*(dis(gen)-0.5); // Each call to dis(gen) generates a new random double
		  // replicate vmod
		  for (int jj=0;jj<ns;jj++) {
			if (jj==ii) {
			  continue;
			} else if (jj<ii && i-addbin*(ii-jj)>=0) {
			  vmod[jj][i-addbin*(ii-jj)] = vmod[ii][i];
			} else if (jj>ii && i+addbin*(jj-ii)<nbin) {
			  vmod[jj][i+addbin*(jj-ii)] = vmod[ii][i];
			}
		  }

		  log_like_try0 = log_likelihood_hmm(vmod,w);
		  log_like_try1 = log_prior(vmod);
		  log_like_try =  log_like_try0 + log_like_try1;
		  double dlog=log_like_try - log_like;
		  double r=dis(gen); // Each call to dis(gen) generates a new random double
		  if (r<exp(dlog/temp)) { // accept move
			v[ii][i] = vmod[ii][i];
			// replicate v
			for (int jj=0;jj<ns;jj++) {
			  if (jj==ii) {
				continue;
			  } else if (jj<ii && i-addbin*(ii-jj)>=0) {
				v[jj][i-addbin*(ii-jj)] = v[ii][i];
			  } else if (jj>ii && i+addbin*(jj-ii)<nbin) {
				v[jj][i+addbin*(jj-ii)] = v[ii][i];
			  }
			}
			naccv[ii] += 1.0;
			naccv_update[ii][i] += 1.0;
			log_like = log_like_try;
			log_like0 = log_like_try0;
			log_like1 = log_like_try1;
		  }
		}

		// Modify W
		double wmod[ns];
		for (int jj=0;jj<ns;jj++) { // copying w to wmod
		  wmod[jj] = w[jj];
		}
		wmod[ii] += dw[ii]*(dis(gen)-0.5); // Each call to dis(gen) generates a new random double
		// replicate wmod
		for (int jj=0;jj<ns;jj++) {
		  if (jj==ii) {
			continue;
		  } else {
			wmod[jj] = wmod[ii];
		  }
		}

		log_like_try0 = log_likelihood_hmm(v,wmod);
		log_like_try1 = log_prior(v);
		log_like_try =  log_like_try0 + log_like_try1;
		double dlog=log_like_try - log_like;
		double r=dis(gen); // Each call to dis(gen) generates a new random double
		if (r<exp(dlog/temp)) { // accept move
		  w[ii] = wmod[ii];
		  // replicate w
		  for (int jj=0;jj<ns;jj++) {
			if (jj==ii) {
			  continue;
			} else {
			  w[jj] = w[ii];
			}
		  }
		  naccw[ii] += 1.0;
		  naccw_update[ii] += 1.0;
		  log_like = log_like_try;
		  log_like0 = log_like_try0;
		  log_like1 = log_like_try1;
		}

      }
      
	  
    } else { // do not replicate (original version)

      for (int ii=0;ii<ns;ii++) {
		
		// Modify V: loop over bins
		double vmod[ns][nbin];
		for (int jj=0;jj<ns;jj++) { // copying v to vmod
		  for (int j=0;j<nbin;j++) { 
			vmod[jj][j] = v[jj][j];
		  }
		}
		for (int i=0;i<nbin;i++) {
		  vmod[ii][i] += dv[ii][i]*(dis(gen)-0.5); // Each call to dis(gen) generates a new random double
		  log_like_try0 = log_likelihood_hmm(vmod,w); 
		  log_like_try = log_like_try0;
		  double dlog=log_like_try - log_like;
		  double r=dis(gen); // Each call to dis(gen) generates a new random double
		  if (r<exp(dlog/temp)) { // accept move
			v[ii][i] = vmod[ii][i];
			naccv[ii] += 1.0;
			naccv_update[ii][i] += 1.0;
			log_like = log_like_try;
			log_like0 = log_like_try0;
		  }
		}
		
		// Modify W
		double wmod[ns];
		for (int jj=0;jj<ns;jj++) { // copying w to wmod
		  wmod[jj] = w[jj];
		}
		wmod[ii] += dw[ii]*(dis(gen)-0.5); // Each call to dis(gen) generates a new random double
		log_like_try0 = log_likelihood_hmm(v,wmod); 
		log_like_try = log_like_try0;
		double dlog=log_like_try - log_like;
		double r=dis(gen); // Each call to dis(gen) generates a new random double
		if (r<exp(dlog/temp)) { // accept move
		  w[ii] = wmod[ii];
		  naccw[ii] += 1.0;
		  naccw_update[ii] += 1.0;
		  log_like = log_like_try;
		  log_like0 = log_like_try0;
		}
		
      }
	  
    }
	
    // MC of rate
    if (mcrate==1 || mcrate==2) {
      double ratefcopy, ratebcopy;
      // ratef
      ratefcopy = ratef_switch;
      ratef_switch += drate[0]*(dis(gen)-0.5);
      if ((ratef_switch<0.0) || (ratef_switch*dt>1.0)) {
		ratef_switch = ratefcopy;
      }
	  log_like_try0 = log_likelihood_hmm(v,w);
	  log_like_try1 = log_prior(v);
	  log_like_try = log_like_try0 + log_like_try1;
      double dlog=log_like_try - log_like;
      double r=dis(gen); // Each call to dis(gen) generates a new random double
      if (r<exp(dlog/temp)) { // accept move
		log_like = log_like_try;
		log_like0 = log_like_try0;
		log_like1 = log_like_try1;
		naccrate_update[0] += 1.0;
      } else {
		ratef_switch = ratefcopy;
      }
	  if (mcrate==1) {
		//rateb
		ratebcopy = rateb_switch;
		rateb_switch += drate[1]*(dis(gen)-0.5);
		if ((rateb_switch<0.0) || (rateb_switch*dt>1.0)) {
		  rateb_switch = ratebcopy;
		}
		log_like_try0 = log_likelihood_hmm(v,w);
		log_like_try1 = log_prior(v);
		log_like_try = log_like_try0 + log_like_try1;
		dlog=log_like_try - log_like;
		r=dis(gen); // Each call to dis(gen) generates a new random double
		if (r<exp(dlog/temp)) { // accept move
		  log_like = log_like_try;
		  log_like0 = log_like_try0;
		  log_like1 = log_like_try1;
		  naccrate_update[1] += 1.0;
		} else {
		  rateb_switch = ratebcopy;
		}
	  }
    }

    // Print intermediate results to screen
    if (imc%10==0 || imc==nmc-1) {
      double ntry1=(imc+1.0)*nbin;
      double ntry2=imc+1.0;
      std::cout <<imc <<" " <<temp <<" " <<log_like <<" " <<log_like0 <<" " <<log_like1 <<" ";
      for (int ins=0;ins<ns;ins++) {
		std::cout <<naccv[ins]/ntry1 <<" " <<naccw[ins]/ntry2 <<" ";
      }
      std::cout <<endl;
    }
    // Print intermediate results to files
    if (imc%10==0 || imc==nmc-1) {
      double ntry1=(imc+1.0)*nbin;
      double ntry2=imc+1.0;
      fo3 <<imc <<" " <<temp <<" " <<log_like <<" " <<log_like0 <<" " <<log_like1 <<" ";
      for (int ins=0;ins<ns;ins++) {
		fo3 <<naccv[ins]/ntry1 <<" " <<naccw[ins]/ntry2 <<" ";
      }
      fo3 <<endl;
      for (int ii=0;ii<ns;ii++) {
		for (int i=0;i<nbin;i++) {
		  fo1 <<v[ii][i] <<" ";
		}
		fo1 <<endl;
		fo2 <<w[ii] <<endl;
      }
      fo5 <<imc <<" " <<ratef_switch <<" " <<rateb_switch <<endl;
    }

    // Update MC move widths
    if (imc==nmc_update_stop) {   // stop adjusting widths
      num_mc_update = 0;
	  // print the final widths to file
	  for (int ii=0;ii<ns;ii++) {
		for (int i=0;i<nbin;i++) {
		  fo6 <<dv[ii][i] <<" ";
		}
		fo6 <<endl;
		fo6 <<dw[ii] <<endl;
	  }
	  fo6 <<drate[0] <<" " <<drate[1] <<endl;
	}
    if (num_mc_update>0) {
      if ((imc+1)%num_mc_update == 0) {
		// adjust widths to target 0.3 acceptance ratio
		for (int ii=0;ii<ns;ii++) {
		  std::cout <<"acc ratio: state" <<ii <<" ";
		  for (int i=0;i<nbin;i++) {
			dv[ii][i] = dv[ii][i]*exp(0.5*(naccv_update[ii][i]/num_mc_update - 0.3));
			std::cout <<naccv_update[ii][i]/num_mc_update <<" ";
			naccv_update[ii][i] = 0.0;
		  }	  
		  dw[ii] = dw[ii]*exp(0.5*(naccw_update[ii]/num_mc_update - 0.3));
		  std::cout <<naccw_update[ii]/num_mc_update <<endl;
		  naccw_update[ii] = 0.0;
		}
		// mcrate
		if (mcrate==1 || mcrate==2) {
		  std::cout <<"acc ratio: rate ";
		  for (int j=0;j<2;j++) {
			drate[j] = drate[j]*exp(0.5*(naccrate_update[j]/num_mc_update - 0.3));
			std::cout <<naccrate_update[j]/num_mc_update <<" ";
			naccrate_update[j] = 0.0;	    
		  }
		  std::cout <<endl;
		}
      }
    }
    
  }

  // eigen values of transition matrix
  //  std::ofstream fo4("eigenvalues.out");
  // recalculate trans_mat
  //transition_matrix(nbin, v[0], w[0]);
  // symmetric transition matrix
  //MatDoub trans_mat_sym(nbin,nbin);
  //for (int i=0; i<nbin; i++) {
  //  for (int j=0; j<nbin; j++) {
  //    trans_mat_sym[i][j] = sqrt(trans_mat[i][j]*trans_mat[j][i]);
  //  }
  //}
  // call Symmeig
  //Symmeig myeig2(trans_mat_sym);
  //for (int i=0;i<nbin;i++) {
  //  fo4 <<myeig2.d[i] <<endl;
  //}
  
  //close output files
  fo1.close();
  fo2.close();
  fo3.close();
  //fo4.close();
  fo5.close();
  fo6.close();

  return 0;
}


int path_optimize(double (&v)[ns][nbin], double (&w)[ns])
{
  std::cout <<"hidden-state path optimization..." <<endl;
  //  input model
  std::cout <<"Input model" << endl;
  for (int i=0;i<ns;i++) {
    for (int j=0;j<nbin;j++) {
      std::cout <<v[i][j] <<" ";
    }
    std::cout <<endl;
    std::cout <<w[i] <<endl;
  }
  // transition matrix
  for (int i=0;i<ns;i++) {
    transition_matrix(nbin, v[i], w[i]);
    for (int j=0;j<ksym;j++) {
      int ibin = j/nbin;
      int jbin = j%nbin;
      transprob[i][j] = trans_mat[jbin][ibin]; // transition prob. ibin -> jbin
      //std::cout <<transprob[i][j] <<" ";
    }
    //std::cout <<endl;
  }
  // Analysis over trajectories
  int ntraj = trajseg.size();
  for (int itraj=0;itraj<ntraj;itraj++) {
    std::cout <<"itraj=" <<itraj <<endl;
    int nt = trajseg[itraj].size();
    int nt2 = nt/lagstep + 1;
    // store xdp
    xdp.clear();
    for (int it2=0;it2<nt2;it2++) {
      int it = it2*lagstep;
      xdp.push_back(trajseg[itraj][it]);
    }
    //
    int nstage = nt2 + 2;
    VecInt nstate(nstage);
    nstate[0] = 1;
    nstate[nstage-1] = 1;
    for (int i=1; i<=nstage-2; i++) {
      nstate[i] = 2;
    }
    VecInt answer(nstage);
    answer = dynpro(nstate,costfunc);
    std::string filename = "pathopt_traj" + std::to_string(itraj) + ".out";
    savepathopt(answer,filename);
  }

  return 0;
}

Doub costfunc(Int jj, Int kk, Int ii)
{
  double c;
  const double cforbid = 1.e5; 
  int nt = xdp.size();
  double x1,x2;
  const double kf=1.0/0.005, kb=1.0/5.0;
  // switching prob
  double p11,p12,p22,p21;
  p12 = 1.0 - exp(-kf*dt); //state 1 -> 2 
  p11 = 1.0 - p12; //state 1 -> 1 
  p21 = 1.0 - exp(-kb*dt); //state 2 -> 1 
  p22 = 1.0 - p21; //state 2 -> 2 
  //
  if ((ii==0)||(ii==nt)) {
    c = 0.0;
  } else {
    x1 = xdp[ii-1];
    x2 = xdp[ii];
    //std::cout <<x1 <<" " <<x2;
    // reactive or not
    int reactive = 0;
    if ((x1>=xswitch[0])&&(x1<=xswitch[1])) {
      reactive = 1;
    }
    // positional transition prob.
    int binnum1, binnum2;
    binnum1 = (int)((x1-xlim_low)/dx);
    binnum2 = (int)((x2-xlim_low)/dx);
    if (binnum1<0) {
      binnum1 = 0;
    } else if (binnum1>=nbin) {
      binnum1 = nbin-1;
    }
    if (binnum2<0) {
      binnum2 = 0;
    } else if (binnum2>=nbin) {
      binnum2 = nbin-1;
    }
    int isymbol = binnum1 * nbin + binnum2;
    double xprob1, xprob2;
    xprob1 = transprob[0][isymbol];
    xprob2 = transprob[1][isymbol];
    //std::cout <<xprob1 <<" " <<xprob2 <<" ";
    //
    if (ii==1) {
      if ((jj==0)&&(kk==0)) {
	c = -log(xprob1);
      } else if ((jj==1)&&(kk==1)) {
	c = -log(xprob2);
      } else {
	c = cforbid;
      }
    } else {
      if ((jj==0)&&(kk==0)) {
	c = -log(xprob1*p11);
      } else if ((jj==1)&&(kk==1)) {
	c = -log(xprob2*p22);
      } else if ((jj==0)&&(kk==1)) {
	if (reactive==1) {
	  c = -log(xprob2*p12);
	} else {
	  c = cforbid;
	}
      } else if ((jj==1)&&(kk==0)) {
	if (reactive==1) {
	  c = -log(xprob1*p21);
	} else {
	  c = cforbid;
	}
      }      
    }
  }
  //std::cout <<c <<" ";
  return c;
}

void savepathopt(const VecInt &v, const std::string f)
{
  std::ofstream fo(f);
  for (int i=1; i<v.size()-1; i++) {
      fo <<v[i] <<endl;
  }
  fo.close();
}


int mc_optimize_hmm_rates(double (&v)[ns][nbin], double (&w)[ns])
{
  //std::cout <<"...mc_optimize_hmm_rates... " <<endl;
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0, 1);

  // basic parameters
  double temp=1.0;
  //int nmc=2000; // # of MC moves
  int num_mc_update=50; // # of moves between width adjustments
  //int nmc_update_stop=1000; // after nmc_update_stop, widths are no longer changed

  double drate[2];
  double naccrate_update[2];
  for (int j=0;j<2;j++) {
    drate[j] = 2.0;
    naccrate_update[j] = 0.0;
  }

  // initial log-likelihood
  double log_like,log_like_try;
  log_like = log_likelihood_hmm(v,w) + log_prior(v);
  std::cout <<"initial log-likelihood: " <<log_like <<endl;

  // open output files
  std::ofstream fo3("loglike.out");
  std::ofstream fo5("rate.out");

  // loop over MC moves
  for (int imc=0;imc<nmc;imc++) {

    double ratefcopy, ratebcopy;
    // ratef
    ratefcopy = ratef_switch;
    ratef_switch += drate[0]*(dis(gen)-0.5);
    if ((ratef_switch<0.0) || (ratef_switch*dt>1.0)) {
	ratef_switch = ratefcopy;
    }
    log_like_try = log_likelihood_hmm(v,w) + log_prior(v);
    double dlog=log_like_try - log_like;
    double r=dis(gen); // Each call to dis(gen) generates a new random double
    if (r<exp(dlog/temp)) { // accept move
      log_like = log_like_try;
      naccrate_update[0] += 1.0;
    } else {
      ratef_switch = ratefcopy;
    }
    //rateb
    ratebcopy = rateb_switch;
    rateb_switch += drate[1]*(dis(gen)-0.5);
    if ((rateb_switch<0.0) || (rateb_switch*dt>1.0)) {
      rateb_switch = ratebcopy;
    }
    log_like_try = log_likelihood_hmm(v,w) + log_prior(v);
    dlog=log_like_try - log_like;
    r=dis(gen); // Each call to dis(gen) generates a new random double
    if (r<exp(dlog/temp)) { // accept move
      log_like = log_like_try;
      naccrate_update[1] += 1.0;
    } else {
      rateb_switch = ratebcopy;
    }

    // Print intermediate results to screen
    if (imc%10==0 || imc==nmc-1) {
      std::cout <<imc <<" " <<log_like <<" ";
      std::cout <<endl;
    }
    // Print intermediate results to files
    if (imc%10==0 || imc==nmc-1) {
      fo3 <<imc <<" " <<log_like <<" ";
      fo3 <<endl;
      fo5 <<imc <<" " <<ratef_switch <<" " <<rateb_switch <<endl;
    }
    // Update MC move widths
    if (imc>=nmc_update_stop) // stop adjusting widths
      num_mc_update = 0;
    if (num_mc_update>0) {
      if ((imc+1)%num_mc_update == 0) {
	// adjust widths to target 0.3 acceptance ratio
	// mcrate
	if (mcrate==1) {
	  std::cout <<"acc ratio: rate ";
	  for (int j=0;j<2;j++) {
	    drate[j] = drate[j]*exp(0.5*(naccrate_update[j]/num_mc_update - 0.3));
	    std::cout <<naccrate_update[j]/num_mc_update <<" ";
	    naccrate_update[j] = 0.0;	    
	  }
	  std::cout <<endl;
	}
      }
    }

  }

  //close output files
  fo3.close();
  fo5.close();

  return 0;
}

void switching_region(const std::vector<double> &trajin, VecInt &sr) 
{
  int nt = trajin.size();
  int nt2 = nt/lagstep + 1;
  for (int it2=0;it2<nt2-1;it2++) {
    int it = it2*lagstep;
    //if ((trajin[it]>=xswitch[0]) && (trajin[it]<=xswitch[1]) && (trajin[it+lagstep]>=xswitch[0]) && (trajin[it+lagstep]<=xswitch[1])) {
    if ((trajin[it+lagstep]>=xswitch[0]) && (trajin[it+lagstep]<=xswitch[1])) {
      sr[it2] = 1;
    } else {
      sr[it2] = 0;
    }
  }
}

