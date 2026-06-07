// CSR null model. alpha(eps)=c0+c1*ln(eps) collapses eps-dependence (<0.5% residual);
// log-logistic SF: SF(Rt) = 1/(1+(Rt/ll_scale)^ll_shape). KS<0.03 on pooled simdata.
const MASTER = { c0:2.031525, c1:0.258273, ll_shape:7.87468, ll_scale:24.05651 };
