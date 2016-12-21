// Adopted from multiagent-mapping-common.
#ifndef DEPTH_SEGMENTATION_TESTING_ENTRYPOINT_H_
#define DEPTH_SEGMENTATION_TESTING_ENTRYPOINT_H_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

// Let the Eclipse parser see the macro.
#ifndef TEST
#define TEST(a, b) int Test_##a##_##b()
#endif

#ifndef TEST_F
#define TEST_F(a, b) int Test_##a##_##b()
#endif

#ifndef TEST_P
#define TEST_P(a, b) int Test_##a##_##b()
#endif

#ifndef TYPED_TEST
#define TYPED_TEST(a, b) int Test_##a##_##b()
#endif

#ifndef TYPED_TEST_P
#define TYPED_TEST_P(a, b) int Test_##a##_##b()
#endif

#ifndef TYPED_TEST_CASE
#define TYPED_TEST_CASE(a, b) int Test_##a##_##b()
#endif

#ifndef REGISTER_TYPED_TEST_CASE_P
#define REGISTER_TYPED_TEST_CASE_P(a, ...) int Test_##a()
#endif

#ifndef INSTANTIATE_TYPED_TEST_CASE_P
#define INSTANTIATE_TYPED_TEST_CASE_P(a, ...) int Test_##a()
#endif

namespace depth_segmentation {

class UnitTestEntryPointBase {
 public:
  virtual ~UnitTestEntryPointBase() {}
  // This function must be inline to avoid linker errors.
  inline int run(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    customInit();
    return RUN_ALL_TESTS();
  }

 private:
  virtual void customInit() = 0;
};

class UnitTestEntryPoint : public UnitTestEntryPointBase {
 public:
  virtual ~UnitTestEntryPoint() {}

 private:
  virtual void customInit() {}
};

}  // namespace depth_segmentation

#define DEPTH_SEGMENTATION_TESTING_ENTRYPOINT           \
  int main(int argc, char** argv) {                     \
    depth_segmentation::UnitTestEntryPoint entry_point; \
    return entry_point.run(argc, argv);                 \
  }
#endif  // DEPTH_SEGMENTATION_TESTING_ENTRYPOINT
