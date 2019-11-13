///////////////////////////////////////////////////////////////////////////////
// File:		  main.cpp
// Revision:	  1
// Date Creation: 04.11.2019
// Last Change:	  04.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  Creates a zoomdrive into a mandelbrot
///////////////////////////////////////////////////////////////////////////////


#include "device_info.h"

#include "ParticleRenderer.h"
#include "CudaPointRenderObject.h"

#include "CudaVertexBuffer.h"

#include "mandelbrot_gpu.h"
#include "mandelbrot_cpu.h"

struct MandelbrotProperties
{
	complex_t const z = complex_t(0.0f, 0.0f);
	//complex_t const c = complex_t(0.74529004, -0.11307502);
	//complex_t const c = complex_t(0.745211200, 0.11307);
	//complex_t c = complex_t(1.506, 0.0);
	//complex_t  c = complex_t(0.367981352, 0.435396403);

	std::vector<complex_t> const cs
	{
		complex_t("0.1ffc4c2'b5e3f8eb'c1479272'195f101d", "0.0000000'008494ab'1b774d8f'84d25fec"),
		complex_t("0.0a30255'36283f32'bd812238'8c7a3efc", "0.0af5e36'09b5fa8a'32bfe8b8'b58fb21d"),
		complex_t("0.0c17afd'92683cb0'3d6e7e79'a0f216b7", "0.0113c2b'd58a89db'64e583b7'96072570")
		
	};
	
	
	//size_t width = 8'192;
	//size_t height = 4'608;
	size_t width = 1920; 
	size_t height = 1080;

	//size_t width = 800;
	//size_t height = 600;

	//size_t width = 180;
	//size_t height = 160;

	//double const zoom = 0.995;
	double const zoom = 0.95;
	//double const zoom = 0.9;
	size_t const iterations = 200;
	double unit = (2.74529004 + 1.25470996) / (width) *8;
};

#include <limits>

int main()
{
	// print info
	printDeviceInfo();


	MandelbrotProperties props;

	//for (size_t i = 0; i < 60 * 60; ++i)
	//{
	//	props.unit = props.unit * props.zoom;
	//}

	// draw
	size_t const width = props.width;
	size_t const height = props.height;
	std::vector<CudaPointRenderObject::Vertex> vertices;
	vertices.resize(width * height);

	for (size_t x = 0; x < width; ++x)
	{
		for (size_t y = 0; y < height; ++y)
		{
			vertices[y * width + x].pos = {
			static_cast<float>(x) / static_cast<float>(width) - static_cast<float>(0.5f),
			static_cast<float>(y) / static_cast<float>(height)* (static_cast<float>(height) / static_cast<float>(width))
				- static_cast<float>(height) / static_cast<float>(width) / 2.0f
			};

			vertices[y * width + x].color = { 
				static_cast<float>(x) / static_cast<float>(width),
				static_cast<float>(y) / static_cast<float>(height),
				1.0f };
		}
	}
	
	auto app = ParticleRenderer::createVulkan();

	auto obj = CudaPointRenderObject::createVulkan();


	size_t vertexBufferSize = vertices.size();
	CudaExternalVertexBuffer<CudaPointRenderObject::Vertex> dp_VertexBuffer;


	bool keyUpPressed = false;
	bool keyDownPressed = false;
	bool keyRightPressed = false;
	bool keyLeftPressed = false;
	bool keyHoldPressed = false;
	double unit = props.unit;
	size_t csIndex = 0;
	complex_t c = props.cs[csIndex];

	auto lbd_device = [&](bool init) {
		// manipulate dp_VertexBuffer
		assert(dp_VertexBuffer.size() == vertices.size());
		assert(sizeof(CudaPointRenderObject::Vertex) == sizeof(Vertex));

		if (init)
		{
			CUDA_CHECK(cudaMemcpy(dp_VertexBuffer.get(), vertices.data(), dp_VertexBuffer.size() * sizeof(CudaPointRenderObject::Vertex), cudaMemcpyHostToDevice));
		}

		//static float count = 0;
		//vertices[0].pos.x = sin(count += 0.001f);

		

		#define cudaEventBlockingSync 0x01

		cudaEvent_t event = { 0 };
		CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
		mandelbrot_gpu(reinterpret_cast<Vertex*>(dp_VertexBuffer.get()), height, width, props.z, c, unit);

		if (keyUpPressed)
		{
			c.imag -= unit*4;
		}
		if (keyDownPressed)
		{
			c.imag += unit*4;
		}
		if (keyLeftPressed)
		{
			c.real += unit*4;
		}
		if (keyRightPressed)
		{
			c.real -= unit*4;
		}
		if (unit < 1e-40)
		{
			unit = props.unit;
			c = props.cs[++csIndex % props.cs.size()];
		}

		if (!keyHoldPressed)
		{
			unit = unit * props.zoom;
		}

		


		CUDA_CHECK(cudaEventRecord(event));
		CUDA_CHECK(cudaEventSynchronize(event));

		//CUDA_CHECK(cudaDeviceSynchronize());



	};


	auto lbd_host = [&](bool init) {
		// manipulate dp_VertexBuffer
		assert(dp_VertexBuffer.size() == vertices.size());
		assert(sizeof(CudaPointRenderObject::Vertex) == sizeof(Vertex));

		static double unit = props.unit;

		mandelbrot_cpu(reinterpret_cast<Vertex*>(vertices.data()), height, width, props.z, c, unit);

		CUDA_CHECK(cudaMemcpy(dp_VertexBuffer.get(), vertices.data(), dp_VertexBuffer.size() * sizeof(CudaPointRenderObject::Vertex), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaDeviceSynchronize());

		unit = unit * props.zoom;
	};


	obj->setVertices(dp_VertexBuffer, vertexBufferSize, lbd_device);
	//obj->setPosition()
	
	app->add(std::move(obj));
	app->setVSync(true);

	
	auto const labda_keyPressed = [&](Key key)
	{
		switch (key)
		{
		case Key::W:
			keyUpPressed = true;
			break;
		case Key::S:
			keyDownPressed = true;
			break;
		case Key::A:
			keyLeftPressed = true;
			break;
		case Key::D:
			keyRightPressed = true;
			break;

		case Key::H:
			keyHoldPressed = true;
			break;

		case Key::P:
			std::cout << "c.real=" << c.real << ", c.imag=" << c.imag << std::endl;
			std::cout <<"unit=" << unit << std::endl;
			break;
		default: 
			break;
		}
	};
	auto const labda_keyReleased = [&](Key key)
	{
		switch (key)
		{
		case Key::W:
			keyUpPressed = false;
			break;
		case Key::S:
			keyDownPressed = false;
			break;
		case Key::A:
			keyLeftPressed = false;
			break;
		case Key::D:
			keyRightPressed = false;
			break;

		case Key::H:
			keyHoldPressed = false;
			break;

		default:
			break;
		}
	};
	app->keyPressed.add(labda_keyPressed);
	app->keyReleased.add(labda_keyReleased);

	app->run();


	return 0;
}