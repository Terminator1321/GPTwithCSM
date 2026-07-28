# Build the GPT-from-scratch project.
#
#   make          -> CPU build (plain g++, no CUDA toolkit required)
#   make cpu      -> same as above, explicit target -> ./gpt_main
#   make cuda     -> GPU-accelerated build (requires nvcc / CUDA toolkit) -> ./gpt_main_cuda
#   make clean    -> remove built binaries
#
# The CUDA build only offloads the two hottest ops in the model
# (Tensor::matmul, used by attention, and LinearLayer's forward/backward,
# used by every projection + the MLP + the LM head) onto the GPU. Everything
# else (embeddings, layer norm, GELU, softmax, the optimizer step, ...)
# still runs on the CPU. Small matmuls fall back to the CPU automatically
# (see kCudaMatmulThreshold / kCudaLinearThreshold) since GPU dispatch
# overhead isn't worth it for tiny layers/short sequences.

CXX      := g++
CXXFLAGS := -std=c++20 -O2 -I.

NVCC       := nvcc
NVCCFLAGS  := -std=c++20 -O2 -I. -DUSE_CUDA -Xcompiler -Wno-deprecated-gpu-targets

SRCS := main.cpp \
        gpt/GPT.cpp \
        gpt/Transformer.cpp \
        gpt/ModelSummary.cpp \
        gpt/Checkpoint.cpp \
        gpt/Layers/Embedding.cpp \
        gpt/Layers/LayerNorms.cpp \
        gpt/Layers/Linear.cpp \
        gpt/Layers/MLP.cpp \
        gpt/Layers/MultiHeadAttention.cpp \
        gpt/Layers/PositionEmbedding.cpp \
        gpt/Layers/ScaledDotProductAttention.cpp \
        gpt/Tensor/Tensor.cpp \
        gpt/NameSpaces/ActivationFunction.cpp \
        gpt/NameSpaces/LOSS.cpp \
        gpt/Optimizers/Adam.cpp \
        gpt/Optimizers/AdamW.cpp \
        gpt/TokenizerLayer/Tokenizer.cpp

CUDA_SRCS := gpt/Tensor/TensorCuda.cu

.PHONY: all cpu cuda clean

all: cpu

cpu: gpt_main

gpt_main: $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o gpt_main

cuda: gpt_main_cuda

gpt_main_cuda: $(SRCS) $(CUDA_SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) $(CUDA_SRCS) -o gpt_main_cuda

clean:
	rm -f gpt_main gpt_main_cuda
