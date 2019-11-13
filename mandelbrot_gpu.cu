///////////////////////////////////////////////////////////////////////////////
// File:		  mandelbrot.cu
// Revision:	  1
// Date Creation: 06.11.2019
// Last Change:	  06.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  create a mandelbrot on the gpu
///////////////////////////////////////////////////////////////////////////////

#include "mandelbrot_gpu.h"

#include <math_extended.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>


__global__ void kernel(Vertex dp_vertexBuffer[], size_t const height, size_t const width,
	complex_t const z, complex_t const c, double const unit)
{
	assert(dp_vertexBuffer != nullptr);

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < height && x < width)
	{
		// Mandelbrot Menge
		auto const res = mandelbrot_iterate_color(z, get_index_value(x, y, width, height, c, unit));

		dp_vertexBuffer[y * width + x].color = {
			static_cast<float>(res.R) / 256.0f,
			static_cast<float>(res.G) / 256.0f,
			static_cast<float>(res.B) / 256.0f 
		};
	}
}


void mandelbrot_gpu(Vertex dp_vertexBuffer[], size_t const height, size_t const width,
	complex_t const& z, complex_t const& c, double const unit)
{
	if (dp_vertexBuffer == nullptr)
	{
		return;
	}


	size_t const block_size = 128;

	unsigned int bigX = static_cast<unsigned int>(ceil_div(width, block_size));
	unsigned int bigY = static_cast<unsigned int>(ceil_div(height, 1));

	unsigned int tibX = static_cast<unsigned int>(block_size);
	unsigned int tibY = static_cast<unsigned int>(1);

	dim3 const big(bigX, bigY);	// blocks in grid
	dim3 const tib(tibX, tibY); // threads in block

	kernel << < big, tib >> > (dp_vertexBuffer, height, width, z, c, unit);
}