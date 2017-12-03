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
PBD_4(float3 *pos, float3 *ppos, bool s1, bool s2, const float dt, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > N) return;

    float3 F_ext = make_float3(0,0,0);
    float3 scale = make_float3(1.0f,1.0f,1.0f);

    float3 _pos = pos[idx];
    float3 _ppos = ppos[idx];
    float3 _vel = scale_float3(1.0f/dt, dif_float3(_pos, _ppos));

    // add F_ext

    // floor fliction
    if (_pos.y <= 0)
    {
        scale = (s2) ? make_float3(0.1f,0.01f,0.1f) : make_float3(1.0f, 0.1f, 1.0f);
        _pos.y = 0;
    }
    // flyaway
    if (s1) F_ext = add_float3(F_ext, make_float3(0, 15.0f, 0));
    // gravity
    F_ext = add_float3(F_ext, make_float3(0, -9.8f, 0));
   

    ppos[idx] = _pos;
    _vel = add_float3(_vel, scale_float3(dt, F_ext));
    pos[idx] = add_float3(_pos, scale_float3(dt, mul_float3(scale, _vel)));

    return;
}
