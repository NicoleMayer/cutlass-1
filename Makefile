all: kernel-call-device

kernel-call-device: kernel-call-device.cu
	nvcc -Iinclude -Iexamples/common -Ibuild/include -Itools/util/include -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
	-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -std=c++14 -gencode=arch=compute_86,code=sm_86 \
	kernel-call-device.cu -o kernel-call-device -O3 -Xptxas -v

clean:
	rm -rf kernel-call-device
