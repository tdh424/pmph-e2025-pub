#ifndef HELPER
#define HELPER

struct U64bits {
    using uint_t = uint64_t;
    using sint_t = int64_t;
    using ubig_t = unsigned __int128;
    using carry_t= uint32_t;
    static const int32_t  bits = 64;
    static const uint_t HIGHEST = 0xFFFFFFFFFFFFFFFF;
};

struct U32bits {
    using uint_t = uint32_t;
    using sint_t = int32_t;
    using ubig_t = uint64_t;
    using carry_t= uint32_t;
    static const int32_t  bits = 32;
    static const uint_t HIGHEST = 0xFFFFFFFF;
};

void validate(const uint64_t N, float* ref, float* res) {
  for(uint64_t i = 0; i < N; i++) {
    if(ref[i] != res[i]) {
      printf("INVALID result at index %ld: refernce is %f result is %f\n", i, ref[i], res[i]);
      exit(0);
    }
  }
  printf("VALID!\n");
}

void initArray(const uint64_t N, float* inp_arr) {
    for(uint64_t i=0; i<N; i++) {
        inp_arr[i] = 2.0 * ( ((float)rand()) / RAND_MAX );
    }
}
#endif // HELPER
