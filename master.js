// CSR null model. alpha(eps)=c0+c1*ln(eps) collapses eps-dependence (<0.5% residual).
// Master SF on Rt = R*eps^alpha(eps) is a SHIFTED INVERSE-GAMMA with integer shape 10
// (= min_samples; it is the a->inf limit of the legacy Beta-Prime(a~46, b~10) fits):
//   SF(Rt) = P(10, y),  y = ig_scale/(Rt - ig_loc),  P = lower regularized gamma.
// Integer shape => exact Poisson sums, no gamma library. Calibrated on simdata
// eps 1.10-1.40: |median_z| <= 0.017, band asymmetry <= 0.014, KS(z) <= 0.009
// (log-logistic master had median_z ~ -0.06: see docs/RCA.md addendum).
const MASTER = { c0:2.031525, c1:0.258273, ig_shape:10, ig_loc:7.5091, ig_scale:157.7035 };

function masterSF(rt){
  if(rt<=MASTER.ig_loc) return 1;
  const y=MASTER.ig_scale/(rt-MASTER.ig_loc);
  if(y>=15){ // body: SF = 1 - exp(-y)*sum_{k=0}^{9} y^k/k!
    let s=1,t=1; for(let k=1;k<10;k++){t*=y/k;s+=t;}
    return 1-Math.exp(-y)*s;
  }
  // tail: SF = exp(-y)*sum_{k=10}^{79} y^k/k!  -- direct series, no cancellation
  let t=Math.exp(10*Math.log(y)-y-15.104412573075516); // ln(10!)
  let s=t; for(let k=11;k<80;k++){t*=y/k;s+=t;}
  return s;
}
function masterCDF(rt){ return 1-masterSF(rt); }
function masterPDF(rt){ // shifted inv-gamma pdf, shape 10: y^10 e^-y / (9! * (rt-loc))
  const u=rt-MASTER.ig_loc; if(u<=0) return 0;
  const y=MASTER.ig_scale/u;
  return Math.exp(10*Math.log(y)-y-12.801827480081469)/u; // ln(9!)
}
