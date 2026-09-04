// Compile-only controls: template indices must remain integral constants, and
// empty/nonascending ranges must not instantiate an uncallable body.
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
using namespace metal;
using namespace mlx::steel;

struct Uncallable {};
struct UnitStep {
  device int* output;
  template <typename Index> void operator()(Index index) const {
    static_assert(Index::value >= 0 && Index::value < 4,
                  "unit-step index must stay in range");
    output[index] = Index::value;
  }
};
struct StridedStep {
  device int* output;
  template <typename Index> void operator()(Index index) const {
    static_assert(Index::value == 2 || Index::value == 4 || Index::value == 6,
                  "strided indices must stay exact");
    output[index + 4] = Index::value;
  }
};

kernel void metal3_const_for_controls(device int* output [[buffer(0)]]) {
  const_for_loop<0, 0, 1>(Uncallable{});
  const_for_loop<4, 2, 1>(Uncallable{});
  const_for_loop<0, 4, 1>(UnitStep{output});
  const_for_loop<2, 7, 2>(StridedStep{output});
}

#ifdef EXPECT_INSTANTIATION_FAILURE
struct RejectIndexTwo {
  template <typename Index> void operator()(Index) const {
    static_assert(Index::value != 2,
                  "negative control must instantiate index two");
  }
};
kernel void metal3_negative_control() {
  const_for_loop<0, 4, 1>(RejectIndexTwo{});
}
#endif
