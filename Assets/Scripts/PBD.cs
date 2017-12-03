using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;

public class PBD : CudaGenerator {

    public Shader _shader;
    private Material _material;

    // 粒子数
    public int NUM_OF_P;
    public int NP = 16;
    // iteration
    [Range(1,30)] public int iter;
    // Time
    private float dt;
    [Range(0.2f, 5.0f)] public float timeScale;

    // 環境変数
    [Range(1.0f, 10.0f)] public float h;

    // 配置
    [Range(0, 1)] public float radius;
    [Range(1, 100)] public int edge_x, edge_z;
    [Range(0, 1)] public float edge_scale;

    // 粒子パラメタ
    private Vector3[] h_pos;
    private Vector3[] h_ppos;
    private float[] h_col;

    // コンピュートバッファ
    ComputeBuffer posBuf;
    ComputeBuffer colBuf;

    // cuda variables
    CudaDeviceVariable<Vector3> d_pos;
    CudaDeviceVariable<Vector3> d_ppos;
    CudaDeviceVariable<Vector3> d_delta_p;
    CudaDeviceVariable<int> d_M_index;
    CudaDeviceVariable<float> d_M;


    public bool S_flyAway;
    public bool S_floorFliction;
    public bool S_saveImages;
    int capture;

    // Use this for initialization
    void Start () {

        InitializeCUDA();
        _material = new Material(_shader);

        dt = Time.deltaTime * timeScale;

        h_pos = new Vector3[NUM_OF_P];
        h_ppos = new Vector3[NUM_OF_P];
        h_col = new float[NUM_OF_P];

        posBuf = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(Vector3)));
        colBuf = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(float)));

        for (int i = 0; i < NUM_OF_P; ++i)
        {
            h_pos[i] = new Vector3(i % edge_x, i / edge_x / edge_z, i / edge_x % edge_z) * edge_scale + new Vector3(0, 5.0f, 0);
            h_ppos[i] = new Vector3(h_pos[i].x + 0.01f, h_pos[i].y, h_pos[i].z);
            h_col[i] = UnityEngine.Random.Range(0.3f, 0.7f);
        }

        d_pos = h_pos;
        d_ppos = h_ppos;
        d_delta_p = h_pos;
        d_M_index = new int[NUM_OF_P * NP];
        d_M = new float[NUM_OF_P * NP];


        int _block = 1024, _grid = (NUM_OF_P + _block - 1) / _block;
        cudaKernel[0].BlockDimensions = new dim3(_block, 1, 1); // max 2048:(2^10, 2, 1)
        cudaKernel[0].GridDimensions = new dim3(_grid, 1, 1); // max (2^31-1, 2^16-1, 2^16-1) 1.8e22個   
        cudaKernel[1].BlockDimensions = new dim3(_block, 1, 1);
        cudaKernel[1].GridDimensions = new dim3(_grid, 1, 1);
        cudaKernel[2].BlockDimensions = new dim3(_block, 1, 1);
        cudaKernel[2].GridDimensions = new dim3(_grid, 1, 1);
        cudaKernel[3].BlockDimensions = new dim3(_block, 1, 1);
        cudaKernel[3].GridDimensions = new dim3(_grid, 1, 1);

        cudaKernel[0].Run(d_pos.DevicePointer, d_M_index.DevicePointer, d_M.DevicePointer, h, NUM_OF_P, NP);
    }

    // Update is called once per frame
    void Update () {

        // cudaKernel[0].Run(d_pos.DevicePointer, d_M_index.DevicePointer, d_M.DevicePointer, h, NUM_OF_P, NP);

        for (int i = 0; i < iter; ++i)
        {
            cudaKernel[1].Run(d_pos.DevicePointer, d_delta_p.DevicePointer, d_M_index.DevicePointer, d_M.DevicePointer, NUM_OF_P, NP);
            cudaKernel[2].Run(d_pos.DevicePointer, d_delta_p.DevicePointer, NUM_OF_P, NP);
        }
        cudaKernel[3].Run(d_pos.DevicePointer, d_ppos.DevicePointer, S_flyAway, S_floorFliction, dt, NUM_OF_P);

        d_ppos.CopyToHost(h_pos);

        posBuf.SetData(h_pos);
        colBuf.SetData(h_col);

        // Print Screen
        if (S_saveImages) ScreenCapture.CaptureScreenshot("images/" + string.Format("{0:00000}", capture) + ".png");
        ++capture;

    }

    void OnDisable()
    {
        d_pos.Dispose();
        d_ppos.Dispose();
        d_delta_p.Dispose();
        d_M_index.Dispose();
        d_M.Dispose();
        posBuf.Release();
        colBuf.Release();
    }

    void OnRenderObject()
    {
        // テクスチャ、バッファをマテリアルに設定

        _material.SetPass(0);
        _material.SetBuffer("posBuf", posBuf);
        _material.SetBuffer("colBuf", colBuf);
        _material.SetFloat("radius", radius);
        _material.SetFloat("Light", 3);
        _material.SetInt("NUM_OF_O", 0);


        // レンダリングを開始
        Graphics.DrawProcedural(MeshTopology.Points, NUM_OF_P);
    }
}
