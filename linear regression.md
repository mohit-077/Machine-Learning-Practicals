// Practical 2: Simple and Multiple Linear Regression   (terminal output only - figures are provided separately)
// Compile : g++ 02_linear_regression.cpp -o p2
// Run     : ./p2      (Windows: .\p2)
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
// 2. SIMPLE + MULTIPLE LINEAR REGRESSION
// ============================================================================

static void p2(){
  header(2,"Simple and Multiple Linear Regression",R"(Model:  y = w0 + w1 x1 + ... + wd xd
 1. Build design matrix X with a leading column of 1s (intercept).
 2. Minimise the sum of squared errors  ||y - Xw||^2.
 3. Solve the Normal Equations  (X^T X) w = X^T y  by Gaussian elimination.
 4. Predict  y_hat = X w.   Evaluate with R^2 and MSE on test data.
 Simple regression uses 1 feature (area); multiple regression uses all 3.)");
  int n=150,ntr=112; Mat X; Vec y;
  for(int i=0;i<n;i++){ double area=randu(5,30),bed=floor(randu(1,5.999)),age=randu(0,30);
    X.push_back({area,bed,age}); y.push_back(4*area+6*bed-1.2*age+15+10*randn()); }
  Mat Xa,Xat,Xm,Xmt; Vec ytr(y.begin(),y.begin()+ntr),yte(y.begin()+ntr,y.end());
  for(int i=0;i<n;i++){ (i<ntr?Xa:Xat).push_back({X[i][0]}); (i<ntr?Xm:Xmt).push_back(X[i]); }
  Vec ws=ols(Xa,ytr),wm=ols(Xm,ytr); Vec ps,pm,qs,qm;
  for(auto&r:Xat) ps.push_back(predLin(ws,r)); for(auto&r:Xmt) pm.push_back(predLin(wm,r));
  for(auto&r:Xa) qs.push_back(predLin(ws,r)); for(auto&r:Xm) qm.push_back(predLin(wm,r));
  cout<<"Dataset : house prices, "<<n<<" samples (train "<<ntr<<", test "<<n-ntr<<")\n";
  cout<<"          features: area (100 sq ft), bedrooms, age (years); target: price (lakh)\n";
  cout<<"True relation used to generate data:\n  price = 15 + 4*area + 6*bedrooms - 1.2*age + noise(sd 10)\n\n";
  cout<<"Simple linear regression  (area only):\n  price = "<<fix(ws[0],3)<<" + "<<fix(ws[1],3)<<" * area\n\n";
  cout<<"Multiple linear regression (area, bedrooms, age):\n  price = "<<fix(wm[0],3)<<" + "<<fix(wm[1],3)<<" * area + "<<fix(wm[2],3)<<" * bedrooms + "<<fix(wm[3],3)<<" * age\n\n";
  cout<<left<<setw(12)<<"Model"<<right<<setw(12)<<"Train R2"<<setw(12)<<"Test R2"<<setw(12)<<"Test MSE"<<setw(12)<<"Test RMSE"<<"\n";
  cout<<left<<setw(12)<<"Simple"<<right<<setw(12)<<fix(r2Of(ytr,qs),4)<<setw(12)<<fix(r2Of(yte,ps),4)<<setw(12)<<fix(mseOf(yte,ps),2)<<setw(12)<<fix(sqrt(mseOf(yte,ps)),2)<<"\n";
  cout<<left<<setw(12)<<"Multiple"<<right<<setw(12)<<fix(r2Of(ytr,qm),4)<<setw(12)<<fix(r2Of(yte,pm),4)<<setw(12)<<fix(mseOf(yte,pm),2)<<setw(12)<<fix(sqrt(mseOf(yte,pm)),2)<<"\n";
  cout<<"\nSample test predictions:\n"<<right<<setw(8)<<"area"<<setw(6)<<"bed"<<setw(6)<<"age"<<setw(10)<<"actual"<<setw(12)<<"simple"<<setw(12)<<"multiple"<<"\n";
  for(int i=0;i<8;i++) cout<<setw(8)<<fix(Xmt[i][0],1)<<setw(6)<<fix(Xmt[i][1],0)<<setw(6)<<fix(Xmt[i][2],1)<<setw(10)<<fix(yte[i],1)<<setw(12)<<fix(ps[i],1)<<setw(12)<<fix(pm[i],1)<<"\n";
}

int main(){
  base=blobs(70,{{2,2},{6.5,3},{4,7.5}},1.2);
  bin =blobs(80,{{2,2},{5.5,5}},1.3);
  p2();
  cout<<"\n(Figures for this practical are provided separately in the 'figures' folder.)\n";
  return 0;
}

<img width="711" height="745" alt="image" src="https://github.com/user-attachments/assets/ab11d949-3c87-413c-95fb-fb9e957a0c15" />
<img width="778" height="602" alt="image" src="https://github.com/user-attachments/assets/9f3169a8-2e77-4449-b6e1-7533e05f4b56" />
<img width="778" height="594" alt="image" src="https://github.com/user-attachments/assets/880302b2-05a8-4d4f-8371-ce397ba145c1" />


