#include "seq.hpp"

#include <algorithm>

namespace matmul {

void band_kernel(const double* __restrict a,
                 const double* __restrict b,
                 double* __restrict c,
                 std::size_t n,
                 std::size_t lda,
                 std::size_t rows) {
    // Loop order ii-kk-jj-i-k-j, the same as on the Rust side.
    //
    // The innermost loop walks C and B SEQUENTIALLY (both pointers step by 1),
    // so the compiler vectorises it with AVX-512 registers. That is why the
    // order is i-k-j and not i-j-k: in the latter case B would be read by
    // columns in the innermost loop and every element would land in a
    // separate cache line.
    //
    // The summation order for every C[i][j] element is increasing k and does
    // NOT depend on the band size or the number of threads. So all variants
    // and both languages produce a bit-for-bit identical result.
    for (std::size_t ii = 0; ii < rows; ii += B) {
        const std::size_t i_max = std::min(ii + B, rows);
        for (std::size_t kk = 0; kk < n; kk += B) {
            const std::size_t k_max = std::min(kk + B, n);
            for (std::size_t jj = 0; jj < n; jj += B) {
                const std::size_t j_max = std::min(jj + B, n);
                const std::size_t width = j_max - jj;
                for (std::size_t i = ii; i < i_max; ++i) {
                    const double* __restrict a_row = a + i * lda;
                    double* __restrict c_row = c + i * lda + jj;
                    for (std::size_t k = kk; k < k_max; ++k) {
                        const double aik = a_row[k];
                        const double* __restrict b_row = b + k * lda + jj;
                        for (std::size_t j = 0; j < width; ++j) {
                            c_row[j] += aik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void multiply(const double* __restrict a,
              const double* __restrict b,
              double* __restrict c,
              std::size_t n,
              std::size_t lda) {
    band_kernel(a, b, c, n, lda, n);
}

}  // namespace matmul
