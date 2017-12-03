//function kernel
__device__ float length(float3 r) {
    return r.x*r.x + r.y*r.y + r.z*r.z;
}
__device__ float3 mul_float3(float3 r1, float3 r2) {
    return make_float3(r1.x * r2.x,  r1.y * r2.y,  r1.z * r2.z);
}
__device__ float3 add_float3(float3 r1, float3 r2) {
    return make_float3(r1.x + r2.x,  r1.y + r2.y,  r1.z + r2.z);
}
__device__ float3 dif_float3(float3 r1, float3 r2) {
    return make_float3(r1.x - r2.x,  r1.y - r2.y,  r1.z - r2.z);
}
__device__ float3 scale_float3(float s, float3 r) {
    r.x *= s;
    r.y *= s;
    r.z *= s;
    return r;
}
__device__ float Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return 315.0f / (64 * PI * pow(h, 9)) * pow(pow(h, 2) - length(r), 3);
}
__device__ float3 Gradient_Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return make_float3(
            r.x * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2),
            r.y * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2),
            r.z * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2));
}
__device__ float Lap_Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return 945.0f / (8 * PI * pow(h, 9)) * (pow(h, 2) - length(r)) * (length(r) - 3 / 4 * (pow(h, 2) - length(r)));
}
__device__ float3 Gradient_Kernel_Spiky(float3 r, float h) {
	float PI = 3.14159;
    float _r = sqrt(length(r));
    float v = -45.0f / (PI * pow(h, 6) * _r) * pow(h - _r, 2);
	return make_float3(r.x*v, r.y*v, r.z*v);
}
__device__ float Lap_Kernel_Viscosity(float3 r, float h) {
	float PI = 3.14159;
	return 45.0f / (PI * pow(h, 5)) * (1 - sqrt(length(r)) / h);
}



//PBF particle struct
struct pPBF {
	float3 pos;
    float3 vel;
    float m;
	float rho;
    float lambda;
	float col;
};

extern "C" __global__ void
PBD_2(float3 *pos, float3 *delta_p, int *M_index, float *M, const int N, const int NP)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > N) return;

    float3 _pos = pos[idx];
    float3 _delta_p = make_float3(0,0,0);
    float Sigma_i = 0, Sigma_j= 0;
    int i;

    // Policy
    // dont use H, and use M for connection
        
    // Sigma_j
    for (i = 0; i < NP; ++i)
    {
        int index = M_index[i + idx * NP];
        if (index == 0) break;
        float3 __pos = pos[index];
        float3 r = dif_float3(_pos, __pos);

        Sigma_j += length(scale_float3(1.0f/sqrt(length(r)), r));
    }

    // delta_p
    for (i = 0; i < NP; ++i)
    {
        int index = M_index[i + idx * NP];
        if (index == 0) break;
        float3 __pos = pos[index];
        float3 r = dif_float3(_pos, __pos);
        float d = M[i + idx * NP];

        Sigma_i = sqrt(length(r)) - d;
        float s = Sigma_i / Sigma_j;
        _delta_p = add_float3(_delta_p, scale_float3(-1.0f*s/sqrt(length(r)), r));
    }

    if (isnan(length(_delta_p))) _delta_p = make_float3(0,0,0);

    delta_p[idx] = _delta_p;

    return;
    
}

/*

    // S
    for (i = 0; i < NP; ++i)
    {
        int index = M[i + idx * NP];
        if (index == 0) break;
        float3 __pos = pos[index];
        float3 r = dif_float3(_pos, __pos);

        Sigma_i += sqrt(length(r)) - d;
        Sigma_j += length(scale_float3(1.0f/sqrt(length(r)), r));
    }

    float s = Sigma_i / Sigma_j;

    if (s >= 1.0f) s = 1.0f;

    // delta_p
    for (i = 0; i < NP; ++i)
    {
        int index = M[i + idx * NP];
        if (index == 0) break;
        float3 __pos = pos[index];

        float3 r = dif_float3(_pos, __pos);
        _delta_p = add_float3(_delta_p, scale_float3(-1.0f*s/sqrt(length(r)), r));
    }

    if (isnan(length(_delta_p))) _delta_p = make_float3(0,0,0);

    delta_p[idx] = _delta_p;

    return;

*/

