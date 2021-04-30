phony:
	nvcc test.cu -o test -O0 -std=c++11 -Wno-deprecated-gpu-targets --debug

