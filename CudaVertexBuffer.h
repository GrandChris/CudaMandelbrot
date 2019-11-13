///////////////////////////////////////////////////////////////////////////////
// File:		  CudaVertexBuffer.h
// Revision:	  1
// Date Creation: 05.11.2019
// Last Change:	  05.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  A vertex buffer improted from vulkan
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cuda_check.h>

#include <cuda_runtime.h>

#include <memory>
#include <cassert>

template<typename TVertex>
class CudaExternalVertexBuffer
{
public:
	// C-Tor
	CudaExternalVertexBuffer() = default;

	CudaExternalVertexBuffer(void * const vulkanMemoryHandle, size_t const size);

	void init(void* const vulkanMemoryHandle, size_t const size);

	// D-Tor
	~CudaExternalVertexBuffer();

	TVertex * get() const;
	size_t size() const;

private:
	cudaExternalMemory_t mCudaExtMemVertexBuffer = nullptr;

	size_t mSize = 0;
	TVertex * dp_VertexBuffer = nullptr;
};






// #######+++++++ Implementation +++++++#######

template<typename TVertex>
inline CudaExternalVertexBuffer<TVertex>::CudaExternalVertexBuffer(void * const vulkanMemoryHandle, size_t const size)
{
	init(vulkanMemoryHandle, size);
}

template<typename TVertex>
inline void CudaExternalVertexBuffer<TVertex>::init(void* const vulkanMemoryHandle, size_t const size)
{
	mSize = size;

	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	cudaExtMemHandleDesc.handle.win32.handle = vulkanMemoryHandle;
	cudaExtMemHandleDesc.size = sizeof(TVertex) * size;

	CUDA_CHECK(cudaImportExternalMemory(&mCudaExtMemVertexBuffer, &cudaExtMemHandleDesc));

	cudaExternalMemoryBufferDesc cudaExtBufferDesc;
	cudaExtBufferDesc.offset = 0;
	cudaExtBufferDesc.size = sizeof(TVertex) * size;
	cudaExtBufferDesc.flags = 0;

	CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&static_cast<void *>(dp_VertexBuffer), mCudaExtMemVertexBuffer, &cudaExtBufferDesc));
}

template<typename TVertex>
inline CudaExternalVertexBuffer<TVertex>::~CudaExternalVertexBuffer()
{
	try
	{
		CUDA_CHECK(cudaDestroyExternalMemory(mCudaExtMemVertexBuffer));
	}
	catch (...)
	{
		assert(false);
	}
}

template<typename TVertex>
inline TVertex* CudaExternalVertexBuffer<TVertex>::get() const
{
	return dp_VertexBuffer;
}

template<typename TVertex>
inline size_t CudaExternalVertexBuffer<TVertex>::size() const
{
	return mSize;
}
