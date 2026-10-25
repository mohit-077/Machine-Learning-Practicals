// Practical 1: Naive Bayes Classifier   (terminal output only - figures are provided separately)
// Compile : g++ 01_naive_bayes.cpp -o p1
// Run     : ./p1      (Windows: .\p1)
#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace std;
typedef vector<double> Vec; typedef vector<Vec> Mat;

// ------------------------------------------------------------------ utilities
static mt19937 rng(42);
static double randn(){ static normal_distribution<double> d(0,1); return d(rng); }
static double randu(double a,double b){ return uniform_real_distribution<double>(a,b)(rng); }
static string num(double v,int p=3){ ostringstream o; o<<setprecision(p)<<v; return o.str(); }
static string fix(double v,int p=3){ ostringstream o; o<<fixed<<setprecision(p)<<v; return o.str(); }
static void say(const string&s){ cout<<s<<"\n"; }
static const char* COL[]={"#e6194b","#3b82f6","#22a35a","#f58231","#911eb4","#808000","#008080","#9a6324"};
static pair<double,double> minmax(const Vec&v){ return {*min_element(v.begin(),v.end()),*max_element(v.begin(),v.end())}; }

struct DS{ Mat X; vector<int> y; int K=0; size_t n()const{return X.size();} };
static DS blobs(int per,vector<array<double,2>> c,double sd){
  DS D; D.K=c.size();
  for(int k=0;k<D.K;k++) for(int i=0;i<per;i++){ D.X.push_back({c[k][0]+sd*randn(),c[k][1]+sd*randn()}); D.y.push_back(k); }
  return D;
}
static DS moons(int per,double noise){
  DS D; D.K=2;
  for(int i=0;i<per;i++){ double t=randu(0,M_PI);
    D.X.push_back({cos(t)+noise*randn(),sin(t)+noise*randn()}); D.y.push_back(0);
    t=randu(0,M_PI);
    D.X.push_back({1-cos(t)+noise*randn(),0.5-sin(t)+noise*randn()}); D.y.push_back(1); }
  return D;
}
static void split(const DS&D,double frac,DS&tr,DS&te){
  vector<int> id(D.n()); iota(id.begin(),id.end(),0); shuffle(id.begin(),id.end(),rng);
  tr=DS(); te=DS(); tr.K=te.K=D.K; size_t nt=frac*D.n();
  for(size_t i=0;i<id.size();i++){ DS&t=(i<nt)?tr:te; t.X.push_back(D.X[id[i]]); t.y.push_back(D.y[id[i]]); }
}
static double acc(const vector<int>&p,const vector<int>&y){ int c=0; for(size_t i=0;i<y.size();i++) c+=(p[i]==y[i]); return (double)c/y.size(); }
static Vec solveLin(Mat A,Vec b){
  int n=b.size();
  for(int c=0;c<n;c++){
    int p=c; for(int r=c+1;r<n;r++) if(fabs(A[r][c])>fabs(A[p][c])) p=r;
    swap(A[c],A[p]); swap(b[c],b[p]);
    for(int r=c+1;r<n;r++){ double f=A[r][c]/A[c][c]; for(int k=c;k<n;k++) A[r][k]-=f*A[c][k]; b[r]-=f*b[c]; }
  }
  Vec x(n); for(int i=n-1;i>=0;i--){ double s=b[i]; for(int k=i+1;k<n;k++) s-=A[i][k]*x[k]; x[i]=s/A[i][i]; }
  return x;
}
// least squares with intercept (w[0]) ; lam = ridge penalty on slopes only
static Vec ols(const Mat&X,const Vec&y,double lam=0){
  int d=X[0].size()+1; Mat A(d,Vec(d,0)); Vec b(d,0);
  for(size_t i=0;i<X.size();i++){ Vec r={1}; r.insert(r.end(),X[i].begin(),X[i].end());
    for(int a=0;a<d;a++){ for(int c=0;c<d;c++) A[a][c]+=r[a]*r[c]; b[a]+=r[a]*y[i]; } }
  for(int a=1;a<d;a++) A[a][a]+=lam; A[0][0]+=1e-12;
  return solveLin(A,b);
}
static double predLin(const Vec&w,const Vec&x){ double s=w[0]; for(size_t j=0;j<x.size();j++) s+=w[j+1]*x[j]; return s; }
static double mseOf(const Vec&y,const Vec&p){ double s=0; for(size_t i=0;i<y.size();i++) s+=(y[i]-p[i])*(y[i]-p[i]); return s/y.size(); }
static double r2Of(const Vec&y,const Vec&p){ double m=0; for(double v:y) m+=v; m/=y.size(); double a=0,b=0; for(size_t i=0;i<y.size();i++){ a+=(y[i]-p[i])*(y[i]-p[i]); b+=(y[i]-m)*(y[i]-m);} return 1-a/b; }

static DS base,bin;   // shared datasets: 3-class blobs and 2-class blobs

// ---------------------------------------------------------- terminal printing
static void header(int no,const string&title,const string&algo){
  cout<<string(70,'=')<<"\n PRACTICAL "<<no<<": "<<title<<"\n"<<string(70,'=')<<"\n\nALGORITHM\n"<<algo<<"\n\n"<<string(70,'-')<<"\nRESULTS\n"<<string(70,'-')<<"\n";
}
static void printConf(const vector<int>&y,const vector<int>&p,int K){
  vector<vector<int> > m(K,vector<int>(K,0)); for(size_t i=0;i<y.size();i++) m[y[i]][p[i]]++;
  cout<<"Confusion matrix (rows = actual, columns = predicted)\n"<<right<<setw(10)<<"";
  for(int k=0;k<K;k++) cout<<setw(10)<<("pred "+to_string(k)); cout<<"\n";
  for(int a=0;a<K;a++){ cout<<setw(10)<<("actual "+to_string(a)); for(int b=0;b<K;b++) cout<<setw(10)<<m[a][b]; cout<<"\n"; }
  cout<<"\nPer-class metrics:\n"<<setw(10)<<"class"<<setw(12)<<"precision"<<setw(10)<<"recall"<<setw(10)<<"F1"<<"\n";
  for(int c=0;c<K;c++){ int tp=m[c][c],fp=0,fn=0; for(int k=0;k<K;k++){ if(k!=c){ fp+=m[k][c]; fn+=m[c][k]; } }
    double pr=(tp+fp)?(double)tp/(tp+fp):0,rc=(tp+fn)?(double)tp/(tp+fn):0,f1=(pr+rc)>0?2*pr*rc/(pr+rc):0;
    cout<<setw(10)<<c<<setw(12)<<fix(pr,3)<<setw(10)<<fix(rc,3)<<setw(10)<<fix(f1,3)<<"\n"; }
}
static void printSamples(const DS&te,const vector<int>&pr){
  cout<<"\nSample predictions (first 8 test points):\n";
  for(int i=0;i<8&&i<(int)te.n();i++) cout<<"  point ("<<fix(te.X[i][0],2)<<", "<<fix(te.X[i][1],2)<<")  predicted="<<pr[i]<<"  actual="<<te.y[i]<<(pr[i]==te.y[i]?"   correct":"   WRONG")<<"\n";
}

// ===================== THIS PRACTICAL =====================

// ============================================================================
// 1. NAIVE BAYES (Gaussian)
// ============================================================================
struct GNB{
  int K,d; Vec prior; Mat mu,var;
  void fit(const DS&D){
    K=D.K; d=D.X[0].size(); prior.assign(K,0); mu.assign(K,Vec(d,0)); var=mu; vector<int> c(K,0);
    for(size_t i=0;i<D.n();i++){ c[D.y[i]]++; for(int j=0;j<d;j++) mu[D.y[i]][j]+=D.X[i][j]; }
    for(int k=0;k<K;k++) for(int j=0;j<d;j++) mu[k][j]/=c[k];
    for(size_t i=0;i<D.n();i++) for(int j=0;j<d;j++){ double t=D.X[i][j]-mu[D.y[i]][j]; var[D.y[i]][j]+=t*t; }
    for(int k=0;k<K;k++){ for(int j=0;j<d;j++) var[k][j]=var[k][j]/c[k]+1e-9; prior[k]=(double)c[k]/D.n(); }
  }
  int predict(const Vec&x)const{
    int best=0; double bv=-1e300;
    for(int k=0;k<K;k++){ double lp=log(prior[k]);
      for(int j=0;j<d;j++) lp+=-0.5*log(2*M_PI*var[k][j])-(x[j]-mu[k][j])*(x[j]-mu[k][j])/(2*var[k][j]);
      if(lp>bv){bv=lp;best=k;} }
    return best;
  }
};

static void p1(){
  header(1,"Naive Bayes Classifier",R"(Training
 1. For every class c compute the prior  P(c) = count(c) / N.
 2. For every class c and feature j compute the mean and variance of x_j.
Prediction for a new point x
 3. For each class c compute
      log P(c|x) = log P(c) + sum_j log N(x_j ; mean_cj , var_cj)
    (features are assumed independent given the class - the 'naive' part)
 4. Output the class with the largest log-posterior.)");
  DS tr,te; split(base,0.75,tr,te); GNB m; m.fit(tr);
  cout<<"Dataset : 3 classes, 2 features, "<<base.n()<<" samples (train "<<tr.n()<<", test "<<te.n()<<")\n\n";
  cout<<"Learned parameters:\n";
  for(int k=0;k<m.K;k++) cout<<"  class "<<k<<"  prior="<<fix(m.prior[k])<<"  mean=("<<fix(m.mu[k][0],2)<<", "<<fix(m.mu[k][1],2)<<")  variance=("<<fix(m.var[k][0],2)<<", "<<fix(m.var[k][1],2)<<")\n";
  vector<int> pr,ptr; for(auto&x:te.X) pr.push_back(m.predict(x)); for(auto&x:tr.X) ptr.push_back(m.predict(x));
  cout<<"\nTrain accuracy : "<<fix(acc(ptr,tr.y)*100,1)<<" %\n";
  cout<<"Test accuracy  : "<<fix(acc(pr,te.y)*100,1)<<" %\n\n";
  printConf(te.y,pr,te.K); printSamples(te,pr);
}

int main(){
  base=blobs(70,{{2,2},{6.5,3},{4,7.5}},1.2);
  bin =blobs(80,{{2,2},{5.5,5}},1.3);
  p1();
  cout<<"\n(Figures for this practical are provided separately in the 'figures' folder.)\n";
  return 0;
}

<img width="682" height="826" alt="image" src="https://github.com/user-attachments/assets/a1d3f4e7-8f05-4323-86d5-b84d20dba411" />
<img width="769" height="595" alt="image" src="https://github.com/user-attachments/assets/a3b4f295-aa66-40ac-8576-a36c77809f56" />

