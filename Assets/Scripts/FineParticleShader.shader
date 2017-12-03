Shader "Custom/FineParticleShader" {
	Properties{
		_MainTex ("Texture (RGB)", 2D) = "white" {}
	}
	SubShader{
		// アルファを使う
		ZWrite On
		Blend SrcAlpha OneMinusSrcAlpha

		Pass{
		CGPROGRAM

		// シェーダーモデルは5.0を指定
#pragma target 5.0

		// シェーダー関数を設定
#pragma vertex vert
#pragma geometry geom
#pragma fragment frag

#include "UnityCG.cginc"

	uniform sampler2D _MainTex;


	struct pPBFP {
		float3 pos;
		float3 vel;
		float m;
		float rho;
		float lambda;
		float col;
	};

	StructuredBuffer<float3> posBuf;
	StructuredBuffer<float> colBuf;
	StructuredBuffer<float> rhoBuf;
	float radius, Light;
	int NUM_OF_O;

	// 頂点シェーダからの出力
	struct v2f {
		float4 pos : SV_POSITION;
		float2 uv : TEXCOORD0;
		float4 col : COLOR;
		float rho : TEXCOORD1;
	};

	// 頂点シェーダ
	v2f vert(uint id : SV_VertexID)
	{
		v2f output;
		output.pos = float4(posBuf[id], 1.0);
		output.uv = float2(0, 0);
		output.rho = rhoBuf[id];
		output.col = float4(0, 1, 0, 1);
		if (colBuf[id] > -1.5)  output.col = float4(1, 0.2, 0.8, 1);
		if (colBuf[id] > -0.5)  output.col = float4(0.8, 1, 0.2, 1);
		if (colBuf[id] > 0.5)  output.col = float4(0.2, 0.7, 1, 1);
		if ((int)id < NUM_OF_O)  output.col = float4(0.2, 0.2, 0.2, 1);
		return output;
	}

	// ジオメトリシェーダ
	[maxvertexcount(4)]
	void geom(point v2f input[1], inout TriangleStream<v2f> outStream)
	{
		float4 pos = input[0].pos;
		float4 col = input[0].col;
		float rho = input[0].rho;

		float4x4 billboardMatrix = UNITY_MATRIX_V;
		billboardMatrix._m03 =
			billboardMatrix._m13 =
			billboardMatrix._m23 =
			billboardMatrix._m33 = 0;

		v2f o;


		//(-1,-1)(-1,1)(1,-1)(1,1)
		for (int x = -1; x < 2; x += 2) {
			for (int y = -1; y < 2; y += 2) {
				float4 cpos = pos + mul(float4(x, y, 0, 0)*radius, billboardMatrix);
				//float4 cpos = pos + mul(billboardMatrix, float4(x, y, 0, 0)*0.3);

				o.pos = mul(UNITY_MATRIX_VP, cpos);
				o.uv = float2(x + 1, y + 1) / 2;
				o.col = float4(1, 1, 1, 1) * col;
				o.rho = rho;

				outStream.Append(o);
			}
		}

		// トライアングルストリップを終了
		outStream.RestartStrip();
	}
	// ピクセルシェーダー
	fixed4 frag(v2f i) : COLOR
	{
		float3 N;
		N.xy = i.uv * 2.0 - 1.0;
		float r2 = dot(N.xy, N.xy);
		if (r2 > 0.96) discard;

		float4 col = i.col;

		float p = (1.0-i.uv.x + i.uv.y) / (Light * 5);
		col = float4(col.x + p, col.y + p, col.z + p, col.w);

		fixed4 tex = tex2D(_MainTex, i.uv);
		tex = float4(tex.x + p, tex.y + p, tex.z + p, tex.w);

		// 色を返す
		// return tex;
		return col;
	}

		ENDCG
	}
	}
}
