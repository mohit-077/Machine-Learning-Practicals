// Practical 3: Polynomial Regression   (terminal output only - figures are provided separately)
// Compile : g++ 03_polynomial_regression.cpp -o p3
// Run     : ./p3      (Windows: .\p3)
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
// 3. POLYNOMIAL REGRESSION
// ============================================================================
static Mat polyFeat(const Vec&x,int deg){ Mat X; for(double v:x){ double z=v/3; Vec r; double p=1; for(int d=1;d<=deg;d++){ p*=z; r.push_back(p);} X.push_back(r);} return X; }

static void p3(){
  header(3,"Polynomial Regression",R"(Model:  y = w0 + w1 x + w2 x^2 + ... + wD x^D
 1. Expand the single input x into features [x, x^2, ..., x^D]
    (x is scaled to [-1,1] for numerical stability).
 2. Treat the expanded features as ordinary multiple linear regression.
 3. Solve the Normal Equations for w.
 4. Repeat for D = 1..10; plot train and test MSE.
 5. Pick the degree where TEST error is minimum (bias-variance trade-off).)");
  int n=60,ntr=30; Vec x,y; for(int i=0;i<n;i++){ double v=randu(-3,3); x.push_back(v); y.push_back(0.5*v*v*v-v*v-2*v+3+2*randn()); }
  Vec xtr(x.begin(),x.begin()+ntr),ytr(y.begin(),y.begin()+ntr),xte(x.begin()+ntr,x.end()),yte(y.begin()+ntr,y.end());
  cout<<"Dataset : y = 0.5x^3 - x^2 - 2x + 3 + noise(sd 2), x in [-3,3]  (train "<<ntr<<", test "<<n-ntr<<")\n\n";
  cout<<right<<setw(8)<<"degree"<<setw(14)<<"train MSE"<<setw(14)<<"test MSE"<<"\n";
  int bestD=1; double bestE=1e18; vector<double> te_;
  for(int d=1;d<=10;d++){ Vec w=ols(polyFeat(xtr,d),ytr,1e-9); Vec a,b; for(auto&r:polyFeat(xtr,d)) a.push_back(predLin(w,r)); for(auto&r:polyFeat(xte,d)) b.push_back(predLin(w,r));
    double e1=mseOf(ytr,a),e2=mseOf(yte,b); te_.push_back(e2); if(e2<bestE){bestE=e2;bestD=d;} }
  for(int d=1;d<=10;d++){ Vec w=ols(polyFeat(xtr,d),ytr,1e-9); Vec a; for(auto&r:polyFeat(xtr,d)) a.push_back(predLin(w,r));
    cout<<setw(8)<<d<<setw(14)<<fix(mseOf(ytr,a),3)<<setw(14)<<fix(te_[d-1],3)<<(d==bestD?"   <-- best (lowest test error)":"")<<"\n"; }
  Vec w3=ols(polyFeat(xtr,3),ytr,1e-9);
  cout<<"\nFitted degree-3 polynomial (converted back to original x):\n  y = "<<fix(w3[0],3)<<" + "<<fix(w3[1]/3,3)<<" x + "<<fix(w3[2]/9,3)<<" x^2 + "<<fix(w3[3]/27,3)<<" x^3\n";
  cout<<"True curve:\n  y = 3 - 2 x - 1 x^2 + 0.5 x^3\n";
  cout<<"\nConclusion: degree 1 under-fits (both errors high); degree "<<bestD<<" has the lowest test error;\nvery high degrees fit the training noise (train error keeps falling, test error rises = over-fitting).\n";
}

int main(){
  base=blobs(70,{{2,2},{6.5,3},{4,7.5}},1.2);
  bin =blobs(80,{{2,2},{5.5,5}},1.3);
  p3();
  cout<<"\n(Figures for this practical are provided separately in the 'figures' folder.)\n";
  return 0;
}

<img width="906" height="700" alt="image" src="https://github.com/user-attachments/assets/0ab96ade-1e69-46d9-bd00-fa9ddf7ce308" />
<img width="777" height="599" alt="image" src="https://github.com/user-attachments/assets/302d37b2-986f-41b1-8633-f9a5fb670975" />
<img width="772" height="594" alt="image" src="https://github.com/user-attachments/assets/8bf4d350-a219-4df6-a2d5-5b19ba1e3f31" />


