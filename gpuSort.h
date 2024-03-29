#pragma once

#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nccl.h>
#include "GpuTypes.h"
#include "Constants.h"

template <typename KeyType, typename ValueType>
class GpuSort {
private:
    unsigned int _items;                                            ///< Total number of items.
    unsigned int _itemStride;                                       ///< Stride of items for sorting.
    int _currentGpu;                                                ///< Index of the current GPU.
    std::vector<cudaStream_t> _streams;                             ///< CUDA streams for each GPU.
    std::vector<std::unique_ptr<GpuBuffer<KeyType>>> _pbKeys;       ///< Buffers for keys on each GPU.
    std::vector<KeyType*> _pKeys;                                   ///< Pointers to keys on each GPU.
    std::vector<std::unique_ptr<GpuBuffer<ValueType>>> _pbValues;   ///< Buffers for values on each GPU.
    std::vector<ValueType*> _pValues;                               ///< Pointers to values on each GPU.
    std::vector<size_t> _tempBytes;                                 ///< Temporary buffer sizes for each GPU.
    std::vector<std::unique_ptr<GpuBuffer<char>>> _pbTemps;         ///< Temporary buffers for each GPU.
    MPI_Comm _mpiComm;                                              ///< MPI communicator.
    std::vector<ncclComm_t> _ncclComms;                             ///< NCCL communicators for each GPU.

public:
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="items">Total number of items.</param>
    /// <param name="mpiComm">MPI communicator.</param>
    GpuSort(unsigned int items, MPI_Comm mpiComm)
        : _items(items),
        _itemStride(((items + 511) >> 9) << 9),
        _currentGpu(0),
        _mpiComm(mpiComm) {
        static_assert(NUM_GPUS > 0, "NUM_GPUS must be greater than 0");

        int gpuCount;
        cudaGetDeviceCount(&gpuCount);
        if (NUM_GPUS > gpuCount) {
            throw std::runtime_error("Not enough GPUs available.");
        }

        _streams.resize(NUM_GPUS);
        _pbKeys.resize(NUM_GPUS);
        _pKeys.resize(NUM_GPUS);
        _pbValues.resize(NUM_GPUS);
        _pValues.resize(NUM_GPUS);
        _tempBytes.resize(NUM_GPUS);
        _pbTemps.resize(NUM_GPUS);
        _ncclComms.resize(NUM_GPUS, nullptr);

        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaStreamCreate(&_streams[i]);

            _pbKeys[i] = std::make_unique<GpuBuffer<KeyType>>(_itemStride * 2);
            _pKeys[i] = _pbKeys[i]->_pDevData;
            _pbValues[i] = std::make_unique<GpuBuffer<ValueType>>(_itemStride * 2);
            _pValues[i] = _pbValues[i]->_pDevData;
            _tempBytes[i] = kInitSort(_items, _pbValues[i].get(), _pbKeys[i].get());
            _pbTemps[i] = std::make_unique<GpuBuffer<char>>(_tempBytes[i]);
        }
        SetActiveGPU(_currentGpu);
    }

    /// <summary>
    /// Destructor.
    /// </summary>
    ~GpuSort() {
        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaStreamDestroy(_streams[i]);
            if (_ncclComms[i]) {
                ncclCommDestroy(_ncclComms[i]);
            }
        }
    }

    /// <summary>
    /// Sets the active GPU.
    /// </summary>
    /// <param name="gpu">Index of the GPU.</param>
    void SetActiveGPU(int gpu) {
        if (gpu >= 0 && gpu < NUM_GPUS) {
            _currentGpu = gpu;
            cudaSetDevice(_currentGpu);
            if (!_ncclComms[_currentGpu]) {
                ncclUniqueId ncclId;
                ncclGetUniqueId(&ncclId);
                ncclCommInitRank(&_ncclComms[_currentGpu], NUM_GPUS, ncclId, _currentGpu);
            }
        }
    }

    /// <summary>
    /// Synchronizes all GPUs.
    /// </summary>
    void SyncAllGPUs() {
        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(_streams[i]);
        }
    }

    /// <summary>
    /// Exchanges data with other GPUs.
    /// </summary>
    void ExchangeDataWithOtherGPUs() {
        int rank, size;
        MPI_Comm_rank(_mpiComm, &rank);
        MPI_Comm_size(_mpiComm, &size);

        int srcRank = (rank + 1) % size;
        int destRank = (rank + size - 1) % size;

        cudaSetDevice(_currentGpu);
        cudaStream_t stream = _streams[_currentGpu];

        cudaError_t cudaErr = cudaDeviceEnablePeerAccess((_currentGpu + 1) % NUM_GPUS, 0);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Error enabling peer access between GPU " << _currentGpu << " and GPU " << (_currentGpu + 1) % NUM_GPUS << ": " << cudaGetErrorString(cudaErr) << std::endl;
            exit(EXIT_FAILURE);
        }

        int prevGpu = (_currentGpu + NUM_GPUS - 1) % NUM_GPUS;
        ncclComm_t prevComm = _ncclComms[prevGpu];
        ncclComm_t currentComm = _ncclComms[_currentGpu];

        ncclGroupStart();
        ncclSend(_pbValues[(_currentGpu + 1) % NUM_GPUS]->_pDevData, _tempBytes[_currentGpu], ncclChar, destRank, prevComm, stream);
        ncclRecv(_pbValues[_currentGpu]->_pDevData, _tempBytes[_currentGpu], ncclChar, srcRank, prevComm, stream);
        ncclGroupEnd();

        MPI_Request sendRequest, recvRequest;
        MPI_Status sendStatus, recvStatus;

        MPI_Isend(_pbValues[(_currentGpu + 1) % NUM_GPUS]->_pDevData, _tempBytes[_currentGpu], MPI_BYTE, destRank, 0, _mpiComm, &sendRequest);
        MPI_Irecv(_pbValues[_currentGpu]->_pDevData, _tempBytes[_currentGpu], MPI_BYTE, srcRank, 0, _mpiComm, &recvRequest);

        cudaDeviceDisablePeerAccess((_currentGpu + 1) % NUM_GPUS);

        cudaStreamSynchronize(stream);
        MPI_Wait(&sendRequest, &sendStatus);
        MPI_Wait(&recvRequest, &recvStatus);
    }

    /// <summary>
    /// Sorts the data on the current GPU.
    /// </summary>
    /// <returns>True if sorting is successful, false otherwise.</returns>
    bool Sort() {
        ExchangeDataWithOtherGPUs();

        return kSort(_items, _pKeys[_currentGpu], _pKeys[_currentGpu ^ 1], _pValues[_currentGpu], _pValues[_currentGpu ^ 1], _pbTemps[_currentGpu]->_pDevData, _tempBytes[_currentGpu]);
    }

    /// <summary>
    /// Gets the buffer for keys on the current GPU.
    /// </summary>
    /// <returns>Pointer to the buffer for keys.</returns>
    [[nodiscard]] GpuBuffer<KeyType>* GetKeyBuffer() const {
        return _pbKeys[_currentGpu].get();
    }

    /// <summary>
    /// Gets the buffer for values on the current GPU.
    /// </summary>
    /// <returns>Pointer to the buffer for values.</returns>
    [[nodiscard]] GpuBuffer<ValueType>* GetValueBuffer() const {
        return _pbValues[_currentGpu].get();
    }

    /// <summary>
    /// Gets the pointer to keys on the current GPU.
    /// </summary>
    /// <returns>Pointer to keys.</returns>
    [[nodiscard]] KeyType* GetKeyPointer() const {
        return _pKeys[_currentGpu];
    }

    /// <summary>
    /// Gets the pointer to values on the current GPU.
    /// </summary>
    /// <returns>Pointer to values.</returns>
    [[nodiscard]] ValueType* GetValuePointer() const {
        return _pValues[_currentGpu];
    }
};
