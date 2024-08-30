#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
using namespace std;

int main()
{
    if (torch::cuda::is_available())
    {
        std::cout << "支持GPU" << std::endl;
    }
    else
    {
        std::cout << "不支持GPU" << std::endl;
    }

    c10::InferenceMode guard;

    torch::Device deviceCUDA(torch::kCUDA, 0);
    torch::Device deviceCPU(torch::kCPU);
    // 加载模型
    torch::jit::script::Module module = torch::jit::load((ROOT_DIR + std::string("checkpoint_epoch5.pt")).c_str(), deviceCUDA);
    module.eval();

    // 预处理图像
    cv::Mat img = cv::imread((ROOT_DIR + std::string("test.png")).c_str());

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGRA2RGB);

    // cv::resize(rgb, rgb, {rgb.cols/2, rgb.rows/2}, 0, 0, cv::INTER_CUBIC); // mismatch with python resize' res

    cv::imshow("input", rgb);

    cv::Mat t = rgb.reshape(1, 1);

    // 两种写法等价
    // 1.
    // torch::Tensor tensor = torch::from_blob(t.data, {1,3, rgb.rows, rgb.cols}, torch::kU8).toType(torch::kFloat32).to(deviceCUDA);
    // 2.
    torch::Tensor tensor = torch::from_blob(t.data, {rgb.rows, rgb.cols, 3}, torch::kU8).toType(torch::kFloat32).to(deviceCUDA).permute({2, 0, 1});
    tensor = tensor.unsqueeze(0);

    tensor = tensor.div(255);

    torch::Tensor output = module.forward({tensor}).toTensor().to(deviceCPU);

    // output = at::sigmoid(output);
    output = torch::softmax(output, 1);

    cv::Mat result = cv::Mat(output.sizes()[2], output.sizes()[3], CV_32FC1);

    memcpy(result.data, output.data_ptr<float>(), sizeof(float) * result.total());

    cv::imshow("res", result);
    cv::waitKey(0);

    result.convertTo(result, CV_8UC1);
    cv::imwrite("result.png", result * 255);

    return 0;
}