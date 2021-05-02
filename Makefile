ADDR := $(shell ./effective_address.sh)
SHARED_DIR=/Users/theevilroot/SharedFun/Cuda/
CUDA_PATH=/Developer/NVIDIA/CUDA-8.0/include/
EXECUTABLE=kernel.co

all: phony

phony:
	rsync -avz src/ theevilroot@$(ADDR):$(SHARED_DIR)
	ssh theevilroot@$(ADDR) 'make -C $(SHARED_DIR) && $(SHARED_DIR)$(EXECUTABLE)' 
 
clean:
	ssh theevilroot@$(ADDR) 'rm -rf $(SHARED_DIR)/*'
