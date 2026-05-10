# HXQ Native Library — Makefile
#
# Build: make        (CPU library + tests)
#        make cuda   (GPU kernel, requires nvcc)
#        make test   (build + run tests)
#        make clean

CC       = gcc
NVCC     = /usr/local/cuda/bin/nvcc
CFLAGS   = -O2 -Wall -Wextra -std=c99 -Iinclude
LDFLAGS  = -lm
CUDA_FLAGS = -O2 -Iinclude

# Source files
SRC      = src/hxq.c
PQ_SRC   = src/polarquant.c
LLOYD_SRC = src/hxq_lloyd.c
CUDA_SRC = src/hxq_cuda.cu
AFFINE_SRC = src/hxq_affine.c
AFFINE_CUDA_SRC = src/hxq_affine_cuda.cu
TEST_SRC = test/test_hxq.c
PQ_TEST_SRC = test/test_polarquant.c
LLOYD_TEST_SRC = test/test_lloyd.c
AFFINE_TEST_SRC = test/test_affine.c

# Output
LIB      = lib/libhxq.a
PQ_LIB   = lib/libpolarquant.a
LLOYD_SO = lib/libhxq_lloyd.so
LLOYD_LIB = lib/libhxq_lloyd.a
TEST_BIN = test/test_hxq
PQ_TEST_BIN = test/test_polarquant
LLOYD_TEST_BIN = test/test_lloyd
AFFINE_LIB = lib/libhxq_affine.a
AFFINE_TEST_BIN = test/test_affine
CUDA_OBJ = lib/hxq_cuda.o
AFFINE_CUDA_OBJ = lib/hxq_affine_cuda.o

.PHONY: all test cuda verify clean polarquant test-polarquant test-lloyd test-affine test-all lloyd

all: $(LIB) $(TEST_BIN) $(PQ_LIB) $(PQ_TEST_BIN) $(LLOYD_SO) $(LLOYD_LIB) $(LLOYD_TEST_BIN) $(AFFINE_LIB) $(AFFINE_TEST_BIN)

# Static library
lib:
	mkdir -p lib

lib/hxq.o: $(SRC) include/hxq.h | lib
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): lib/hxq.o
	ar rcs $@ $^

# Test binary
$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) $< -Llib -lhxq $(LDFLAGS) -o $@

# Verification against Python reference
VERIFY_BIN = test/verify_against_python
$(VERIFY_BIN): test/verify_against_python.c $(LIB)
	$(CC) $(CFLAGS) $< -Llib -lhxq $(LDFLAGS) -o $@

# PolarQuant static library
lib/polarquant.o: $(PQ_SRC) include/polarquant.h | lib
	$(CC) $(CFLAGS) -c $< -o $@

$(PQ_LIB): lib/polarquant.o
	ar rcs $@ $^

polarquant: $(PQ_LIB)

# PolarQuant test binary
$(PQ_TEST_BIN): $(PQ_TEST_SRC) $(PQ_LIB)
	$(CC) $(CFLAGS) $< -Llib -lpolarquant $(LDFLAGS) -o $@

# PolarQuant verification bridge (for Python comparison)
PQ_BRIDGE = test/verify_polarquant_bridge
$(PQ_BRIDGE): test/verify_polarquant_bridge.c $(PQ_LIB)
	$(CC) $(CFLAGS) $< -Llib -lpolarquant $(LDFLAGS) -o $@

# Run PolarQuant tests
test-polarquant: $(PQ_TEST_BIN)
	@echo ""
	@echo "Running PolarQuant tests..."
	@echo ""
	@./$(PQ_TEST_BIN)

# Verify PolarQuant C vs Python
verify-polarquant: $(PQ_BRIDGE)
	@echo ""
	@echo "Verifying PolarQuant C vs Python..."
	@echo ""
	@python3 test/verify_polarquant.py --bridge $(PQ_BRIDGE)

# Lloyd's reassignment shared library (for ctypes from Python)
lib/hxq_lloyd.o: $(LLOYD_SRC) include/hxq_lloyd.h | lib
	$(CC) -O3 -march=native -fopenmp -Wall -Wextra -std=c99 -Iinclude -fPIC -c $< -o $@

$(LLOYD_SO): lib/hxq_lloyd.o
	$(CC) -O3 -march=native -fopenmp -shared -o $@ $< -lm

$(LLOYD_LIB): lib/hxq_lloyd.o
	ar rcs $@ $^

lloyd: $(LLOYD_SO) $(LLOYD_LIB)

# Lloyd's test binary
$(LLOYD_TEST_BIN): $(LLOYD_TEST_SRC) $(LLOYD_LIB)
	$(CC) -O3 -march=native -fopenmp -Wall -Wextra -std=c99 -Iinclude $< -Llib -lhxq_lloyd $(LDFLAGS) -o $@

# Run Lloyd's tests
test-lloyd: $(LLOYD_TEST_BIN)
	@echo ""
	@echo "Running Lloyd's reassignment tests..."
	@echo ""
	@./$(LLOYD_TEST_BIN)

# Affine group quantization static library
lib/hxq_affine.o: $(AFFINE_SRC) include/hxq_affine.h | lib
	$(CC) $(CFLAGS) -c $< -o $@

$(AFFINE_LIB): lib/hxq_affine.o
	ar rcs $@ $^

# Affine test binary (links both affine + main hxq for meta-kernel test)
$(AFFINE_TEST_BIN): $(AFFINE_TEST_SRC) $(AFFINE_LIB) $(LIB)
	$(CC) $(CFLAGS) $< -Llib -lhxq_affine -lhxq $(LDFLAGS) -o $@

# Run affine tests
test-affine: $(AFFINE_TEST_BIN)
	@echo ""
	@echo "Running HXQ affine group quantization tests..."
	@echo ""
	@./$(AFFINE_TEST_BIN)

# Run all tests
test-all: test test-polarquant test-lloyd test-affine

# CUDA kernels (optional)
cuda: $(CUDA_OBJ) $(AFFINE_CUDA_OBJ)

$(CUDA_OBJ): $(CUDA_SRC) include/hxq.h | lib
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

$(AFFINE_CUDA_OBJ): $(AFFINE_CUDA_SRC) include/hxq_affine.h | lib
	$(NVCC) $(CUDA_FLAGS) -DHXQ_HAVE_CUDA -c $< -o $@

# Affine CUDA shared library (for ctypes/Python integration)
AFFINE_CUDA_SO = lib/libhxq_affine_cuda.so
$(AFFINE_CUDA_SO): $(AFFINE_CUDA_SRC) include/hxq_affine.h | lib
	$(NVCC) $(CUDA_FLAGS) -DHXQ_HAVE_CUDA --shared -Xcompiler -fPIC $< -o $@

cuda-so: $(AFFINE_CUDA_SO)

# Affine CUDA test binary
AFFINE_CUDA_TEST_SRC = test/test_affine_cuda.cu
AFFINE_CUDA_TEST_BIN = test/test_affine_cuda

$(AFFINE_CUDA_TEST_BIN): $(AFFINE_CUDA_TEST_SRC) $(AFFINE_CUDA_OBJ) $(AFFINE_LIB)
	$(NVCC) $(CUDA_FLAGS) -DHXQ_HAVE_CUDA $< lib/hxq_affine_cuda.o -Llib -lhxq_affine -lm -o $@

test-affine-cuda: $(AFFINE_CUDA_TEST_BIN)
	@echo ""
	@echo "Running HXQ affine CUDA tests..."
	@echo ""
	@./$(AFFINE_CUDA_TEST_BIN)

# Run tests
test: $(TEST_BIN)
	@echo ""
	@echo "Running HXQ native tests..."
	@echo ""
	@./$(TEST_BIN)

# Verify against Python-exported tensor
verify: $(VERIFY_BIN)
	@echo ""
	@echo "Run export_tensor.py on GPU box first, then:"
	@echo "  ./test/verify_against_python test_meta.json"
	@echo ""

clean:
	rm -f lib/*.o lib/*.a lib/*.so $(TEST_BIN) $(PQ_TEST_BIN) $(PQ_BRIDGE) $(AFFINE_TEST_BIN) $(LLOYD_TEST_BIN) $(AFFINE_CUDA_TEST_BIN)
	rm -rf lib
