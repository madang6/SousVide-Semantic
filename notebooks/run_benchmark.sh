TRT_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt_libs"
CUDNN_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TRT_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH"
exec python "$@"