#ifndef BM3D_PY
#define BM3D_PY

float* bm3d_threshold_colored_interface(double* z, float thr3D, int P, int nm1, int nm2,
										int sz1, int sz2, float thrClose, int searchWinSize,
										float* fMatN1, float* iMatN1, float* arbMat, float* arbMatInv,
										float* sigmas, double* WIN,
										float* PSD, int Nf, int Kin, float gamma, int channelCount,
										int* preBlockMatches);

float* bm3d_wiener_colored_interface(double* Bn, float* AVG, int P, int nm1, int nm2,
									    int sz1, int sz2, float thrClose, int searchWinSize,
										float* fMatN1, float* iMatN1, float* arbMat, float* arbMatInv,
										float* sigmas, double* WIN,
										float* PSD, int Nf, int Kin, int channelCount,
										int* preBlockMatches);

#endif