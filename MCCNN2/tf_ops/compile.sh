rm build/*
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/compute_keys.cu -o build/compute_keys.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/build_grid_ds.cu -o build/build_grid_ds.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/find_ranges_grid_ds.cu -o build/find_ranges_grid_ds.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/count_neighbors.cu -o build/count_neighbors.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/elem_wise_min.cu -o build/elem_wise_min.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/scan_alg.cu -o build/scan_alg.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/store_neighbors.cu -o build/store_neighbors.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/compute_pdf.cu -o build/compute_pdf.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/count_unique_keys.cu -o build/count_unique_keys.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/store_unique_keys.cu -o build/store_unique_keys.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/pooling_avg.cu -o build/pooling_avg.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/count_pooling_pd.cu -o build/count_pooling_pd.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/store_pooled_pts.cu -o build/store_pooled_pts.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/basis/basis_utils.cu -o build/basis_utils.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/basis/basis_proj.cu -o build/basis_proj.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/basis/basis_proj_grads.cu -o build/basis_proj_grads.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/basis/basis_kp.cu -o build/basis_kp.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/basis/basis_hproj.cu -o build/basis_hproj.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  cu/src/basis/basis_hproj_bilateral.cu -o build/basis_hproj_bilateral.cu.o -Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++-4.8 -std=c++11 build/compute_keys.cu.o  build/build_grid_ds.cu.o  build/find_ranges_grid_ds.cu.o  build/count_neighbors.cu.o  build/elem_wise_min.cu.o  build/scan_alg.cu.o  build/store_neighbors.cu.o  build/compute_pdf.cu.o  build/count_unique_keys.cu.o  build/store_unique_keys.cu.o  build/pooling_avg.cu.o  build/count_pooling_pd.cu.o  build/store_pooled_pts.cu.o  build/basis_utils.cu.o  build/basis_proj.cu.o  build/basis_proj_grads.cu.o  build/basis_kp.cu.o  build/basis_hproj.cu.o  build/basis_hproj_bilateral.cu.o  cc/src/tf_gpu_device.cpp  cc/src/compute_keys.cpp  cc/src/build_grid_ds.cpp  cc/src/find_neighbors.cpp  cc/src/compute_pdf.cpp  cc/src/pooling.cpp  cc/src/basis_proj.cpp -o build/MCCNN2.so -shared -fPIC -Icc/header -Icu/header -I/home/michael/.local/lib/python3.6/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -I/usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -L/home/michael/.local/lib/python3.6/site-packages/tensorflow -l:libtensorflow_framework.so.2 -O2
