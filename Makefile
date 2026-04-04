# HXQ Native Library — Makefile
#
# Build: make        (CPU library + tests)
#        make cuda   (GPU kernel, requires nvcc)
#        make test   (build + run tests)
#        make clean

CC       = gcc
NVCC     = nvcc
CFLAGS   = -O2 -Wall -Wextra -std=c99 -Iinclude
LDFLAGS  = -lm
CUDA_FLAGS = -O2 -Iinclude

# Source files
SRC      = src/hxq.c
CUDA_SRC = src/hxq_cuda.cu
TEST_SRC = test/test_hxq.c

# Output
LIB      = lib/libhxq.a
TEST_BIN = test/test_hxq
CUDA_OBJ = lib/hxq_cuda.o

.PHONY: all test cuda verify clean

all: $(LIB) $(TEST_BIN)

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

# CUDA kernel (optional)
cuda: $(CUDA_OBJ)

$(CUDA_OBJ): $(CUDA_SRC) include/hxq.h | lib
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

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
	rm -f lib/*.o lib/*.a $(TEST_BIN)
	rm -rf lib
