#include <opencv2/opencv.hpp>
#include "common/arg_parsing.hpp"
#include "common/cv_cpp_utils.hpp"
#include "yolov8_app/yolov8_seg_cuda/yolov8_seg.hpp"
#include<numeric>

void trt_cuda_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer; // 创建
    tensorrt_infer::yolov8_cuda::YOLOv8Seg yolov8_obj;
    yolov8_obj.initParameters(s->model_path, s->score_thr);

    // 判断图片路径是否存在
    if (!ai::utils::file_exist(s->image_path))
    {
        INFO("Error: image path is not exist!!!");
        exit(0);
    }

    cudaEvent_t start, stop;
	float esp_time_gpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start, 0);// start

    // 加载要推理的数据
    std::vector<cv::Mat> images; // org images shape: [1920, 1080, 3]
    for (int i = 0; i < s->batch_size; i++)
        images.push_back(cv::imread(s->image_path));
    std::vector<ai::cvUtil::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), ai::cvUtil::cvimg_trans_func);

    cudaEventRecord(stop, 0);// stop
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&esp_time_gpu, start, stop);
	printf("Time for the imread: %f ms\n", esp_time_gpu);

    vector<float> mt1, mt2, mt3, mt4,mt5,mt6;
    // 模型预热，如果要单张推理，请调用yolov8_obj.forward
    for (int i = 0; i < s->number_of_warmup_runs; ++i)
        auto warmup_batched_result = yolov8_obj.forwards(yoloimages,mt4,mt5,mt6);

    ai::cvUtil::BatchSegBoxArray batched_result;
    
    // 模型推理
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
        batched_result = yolov8_obj.forwards(yoloimages,mt1,mt2,mt3);
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    float pre_mean   = accumulate(mt1.begin(), mt1.end(), 0.) / s->loop_count;
    float infer_mean = accumulate(mt2.begin(), mt2.end(), 0.) / s->loop_count;
    float post_mean  = accumulate(mt3.begin(), mt3.end(), 0.) / s->loop_count;
    std::cout << "pre: " << pre_mean << ",  infer:" << infer_mean << ",  post: " << post_mean << ", total: " <<pre_mean+infer_mean+post_mean<< "ms"<<std::endl;

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        ai::cvUtil::draw_batch_segment(images, batched_result, s->output_dir, s->classlabels, 200,  800);
        // ai::cvUtil::draw_batch_segment(images, batched_result, s->output_dir, s->classlabels, 160,  640);
        // ai::cvUtil::draw_batch_segment(images, batched_result, s->output_dir, s->classlabels, 120,  480);
        // ai::cvUtil::draw_batch_segment(images, batched_result, s->output_dir, s->classlabels, 56,  224);

    }
}

int main(int argc, char *argv[])
{
    ai::arg_parsing::Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        INFO("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    ai::arg_parsing::printArgs(&s);

    CHECK(cudaSetDevice(s.device_id)); // 设置你用哪块gpu
    trt_cuda_inference(&s);            // tensorrt的gpu版本推理：模型前处理和后处理都是使用cuda实现
    return RETURN_SUCCESS;
}