#include <glog/logging.h>
#include <gtest/gtest.h>

#include <opencv2/core/core.hpp>

#ifndef TEST
#define TEST(a, b) int Test_##a##_##b()
#endif

class ShitTest : public ::testing::Test {
 protected:
  ShitTest() {}
  virtual ~ShitTest() {}
  virtual void SetUp() {}
};

TEST_F(ShitTest, TestThisShit) {
  cv::Mat cov_mat_shit(3, 3, CV_32FC1);
  cov_mat_shit.setTo(0.0f);

  // cv::Mat eigenvalues;
  // cv::Mat eigenvectors;
  // cv::Mat covariance(3, 3, CV_32FC1, 0.0f);
  // covariance.at<float>(0,0) = 1.0f;
  // covariance.at<float>(1,1) = 1.0f;
  // covariance.at<float>(2,2) = 1.0f;
  // cv::eigen(covariance, eigenvalues, eigenvectors);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  return RUN_ALL_TESTS();
}
