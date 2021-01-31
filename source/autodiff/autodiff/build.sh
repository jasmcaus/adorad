# g++ -std=c++2a -O0 -g \

g++ -std=c++2a -Ofast -ffast-math -march=native \
    -fconcepts \
    -I/opt/miniconda3/envs/xtensor/include/ \
    -I/opt/miniconda3/envs/xtensor/include/xtensor/ \
    -I/opt/miniconda3/envs/xtensor/include/xtensor-blas/ \
    -Isrc \
    main.cpp \
    -L/opt/miniconda3/envs/xtensor/lib \
    -lOpenImageIO \
    -lcblas \
    -lfmt \
    -o main
