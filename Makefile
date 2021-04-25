phony:
	nvcc test.cu -o test -O1 -std=c++11 -Wno-deprecated-gpu-targets

