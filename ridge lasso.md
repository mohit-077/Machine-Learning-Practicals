// Practical 4: Lasso and Ridge Regression   (terminal output only - figures are provided separately)
// Compile : g++ 04_ridge_lasso.cpp -o p4
// Run     : ./p4      (Windows: .\p4)
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
// 4. RIDGE & LASSO
// ============================================================================
static Vec ridgeFit(const Mat&Z,const Vec&yc,double lam){
  int P=Z[0].size(); Mat A(P,Vec(P,0)); Vec b(P,0);
  for(size_t i=0;i<Z.size();i++) for(int a=0;a<P;a++){ for(int c=0;c<P;c++) A[a][c]+=Z[i][a]*Z[i][c]; b[a]+=Z[i][a]*yc[i]; }
  for(int a=0;a<P;a++) A[a][a]+=lam; return solveLin(A,b);
}
static Vec lassoFit(const Mat&Z,const Vec&yc,double lam){
  int n=Z.size(),P=Z[0].size(); Vec w(P,0),r=yc;
  for(int sweep=0;sweep<300;sweep++) for(int j=0;j<P;j++){
    double rho=0,zz=0; for(int i=0;i<n;i++){ rho+=Z[i][j]*(r[i]+w[j]*Z[i][j]); zz+=Z[i][j]*Z[i][j]; } rho/=n; zz/=n;
    double nw=(rho>lam)?(rho-lam)/zz:(rho<-lam?(rho+lam)/zz:0);
    for(int i=0;i<n;i++) r[i]-=(nw-w[j])*Z[i][j]; w[j]=nw; }
  return w;
}

static void p4(){
  header(4,"Lasso and Ridge Regression",R"(Standardise all features, centre y.
RIDGE (L2):  minimise ||y - Xw||^2 + lambda * sum w_j^2
   closed form:  w = (X^T X + lambda I)^-1 X^T y
LASSO (L1):  minimise (1/2n)||y - Xw||^2 + lambda * sum |w_j|
   no closed form -> coordinate descent:
     repeat until converged, for each feature j:
        rho_j = (1/n) * sum_i x_ij * (partial residual_i)
        w_j   = soft_threshold(rho_j, lambda) = sign(rho_j) * max(|rho_j| - lambda, 0)
Sweep lambda over a log grid -> coefficient paths.
Choose lambda with the lowest validation/test MSE.)");
  int n=80,nt=60,P=8; Vec tb={3,-2,0,0,1.5,0,0,0};
  auto gen=[&](int m,Mat&X,Vec&y){ for(int i=0;i<m;i++){ double c=randn(); Vec r(P); double t=0; for(int j=0;j<P;j++){ r[j]=randn()+0.6*c; t+=tb[j]*r[j]; } X.push_back(r); y.push_back(t+2*randn()); } };
  Mat Xtr,Xte; Vec ytr,yte; gen(n,Xtr,ytr); gen(nt,Xte,yte);
  Vec mu(P,0),sd(P,0); double ym=0; for(double v:ytr) ym+=v; ym/=n;
  for(int j=0;j<P;j++){ for(auto&r:Xtr) mu[j]+=r[j]; mu[j]/=n; for(auto&r:Xtr) sd[j]+=(r[j]-mu[j])*(r[j]-mu[j]); sd[j]=sqrt(sd[j]/n); }
  Mat Z=Xtr,Zt=Xte; for(auto&r:Z) for(int j=0;j<P;j++) r[j]=(r[j]-mu[j])/sd[j]; for(auto&r:Zt) for(int j=0;j<P;j++) r[j]=(r[j]-mu[j])/sd[j];
  Vec yc=ytr; for(double&v:yc) v-=ym;
  auto tmse=[&](const Vec&w){ double s=0; for(int i=0;i<nt;i++){ double p=ym; for(int j=0;j<P;j++) p+=w[j]*Zt[i][j]; s+=(yte[i]-p)*(yte[i]-p);} return s/nt; };
  int G=30; Vec lxR,lxL,mR,mL; vector<Vec> wR,wL;
  for(int g=0;g<G;g++){
    double lr=-1+4.5*g/(G-1), ll=-2.5+3.2*g/(G-1); lxR.push_back(lr); lxL.push_back(ll);
    Vec a=ridgeFit(Z,yc,pow(10,lr)),b=lassoFit(Z,yc,pow(10,ll)); wR.push_back(a); wL.push_back(b); mR.push_back(tmse(a)); mL.push_back(tmse(b)); }
  int bR=min_element(mR.begin(),mR.end())-mR.begin(),bL=min_element(mL.begin(),mL.end())-mL.begin();
  Vec wO=ridgeFit(Z,yc,1e-8);
  cout<<"Dataset : "<<P<<" correlated features, train "<<n<<", test "<<nt<<"; only x1, x2, x5 truly matter\n";
  cout<<"True coefficients: 3, -2, 0, 0, 1.5, 0, 0, 0   (noise sd 2)\n\n";
  cout<<"Lambda search (test MSE; every 3rd value shown):\n";
  cout<<right<<setw(14)<<"log10(lam_R)"<<setw(12)<<"Ridge MSE"<<setw(16)<<"log10(lam_L)"<<setw(12)<<"Lasso MSE"<<setw(16)<<"Lasso non-zero"<<"\n";
  for(int g=0;g<G;g+=3){ int nz=0; for(int j=0;j<P;j++) if(fabs(wL[g][j])>1e-12) nz++;
    cout<<setw(14)<<fix(lxR[g],2)<<setw(12)<<fix(mR[g],3)<<setw(16)<<fix(lxL[g],2)<<setw(12)<<fix(mL[g],3)<<setw(16)<<nz<<"\n"; }
  cout<<"\nBest Ridge : lambda = "<<fix(pow(10,lxR[bR]),3)<<"   test MSE = "<<fix(mR[bR],3)<<"\n";
  cout<<"Best Lasso : lambda = "<<fix(pow(10,lxL[bL]),4)<<"   test MSE = "<<fix(mL[bL],3)<<"\n";
  cout<<"Plain OLS  :                      test MSE = "<<fix(tmse(wO),3)<<"\n\n";
  cout<<"Coefficients (original scale):\n"<<right<<setw(8)<<"feature"<<setw(9)<<"true"<<setw(10)<<"OLS"<<setw(10)<<"Ridge"<<setw(10)<<"Lasso"<<"\n";
  int zeros=0;
  for(int j=0;j<P;j++){ double lz=wL[bL][j]/sd[j]; if(fabs(lz)<1e-12) zeros++;
    cout<<setw(7)<<"x"+to_string(j+1)<<setw(9)<<fix(tb[j],2)<<setw(10)<<fix(wO[j]/sd[j],3)<<setw(10)<<fix(wR[bR][j]/sd[j],3)<<setw(10)<<(fabs(lz)<1e-12?string("0 (dropped)"):fix(lz,3))<<"\n"; }
  cout<<"\nLasso set "<<zeros<<" coefficient(s) exactly to zero (feature selection); Ridge only shrinks them towards zero.\n";
}

int main(){
  base=blobs(70,{{2,2},{6.5,3},{4,7.5}},1.2);
  bin =blobs(80,{{2,2},{5.5,5}},1.3);
  p4();
  cout<<"\n(Figures for this practical are provided separately in the 'figures' folder.)\n";
  return 0;
}
<img width="775" height="908" alt="image" src="https://github.com/user-attachments/assets/5a86890f-2f0a-418d-9a0e-4efeb09ae108" />
<img width="774" height="597" alt="image" src="https://github.com/user-attachments/assets/be8f130e-1a84-4df6-9bf8-4a8dcfbaedc9" />
<img width="776" height="596" alt="image" src="https://github.com/user-attachments/assets/ce2d4b63-bd88-435c-9c44-684d0bfcdf02" />
<img width="772" height="600" alt="image" src="https://github.com/user-attachments/assets/2f226d18-19ec-429f-809b-91ba6424c0b9" />



