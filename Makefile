all: test-invoke

test-invoke: test.cu
	nvcc -Iinclude -Iexamples/common -Ibuild/include -Itools/util/include -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
	-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -std=c++14 -gencode=arch=compute_86,code=sm_86 \
	test.cu -o test-invoke -O3 -Xptxas -v -rdc=true

clean:
	rm -rf test-invoke
