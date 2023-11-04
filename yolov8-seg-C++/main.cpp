//
// Created by ubuntu on 2/8/23.
//
#include "chrono"
#include "yolov8-seg.hpp"
#include "opencv2/opencv.hpp"

#include <string> 
using namespace std;

const std::vector<std::string> CLASS_NAMES = {"rails"};
const std::vector<std::vector<unsigned int>> COLORS = {{ 0, 114, 189 }};
// const std::vector<std::string> CLASS_NAMES = {
// 	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
// 	"train", "truck", "boat", "traffic light", "fire hydrant",
// 	"stop sign", "parking meter", "bench", "bird", "cat",
// 	"dog", "horse", "sheep", "cow", "elephant",
// 	"bear", "zebra", "giraffe", "backpack", "umbrella",
// 	"handbag", "tie", "suitcase", "frisbee", "skis",
// 	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
// 	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
// 	"cup", "fork", "knife", "spoon", "bowl",
// 	"banana", "apple", "sandwich", "orange", "broccoli",
// 	"carrot", "hot dog", "pizza", "donut", "cake",
// 	"chair", "couch", "potted plant", "bed", "dining table",
// 	"toilet", "tv", "laptop", "mouse", "remote",
// 	"keyboard", "cell phone", "microwave", "oven",
// 	"toaster", "sink", "refrigerator", "book", "clock", "vase",
// 	"scissors", "teddy bear", "hair drier", "toothbrush" };

// const std::vector<std::vector<unsigned int>> COLORS = {
// 	{ 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
// 	{ 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
// 	{ 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
// 	{ 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
// 	{ 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
// 	{ 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
// 	{ 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
// 	{ 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
// 	{ 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
// 	{ 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
// 	{ 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
// 	{ 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
// 	{ 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
// 	{ 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
// 	{ 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
// 	{ 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
// 	{ 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
// 	{ 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
// 	{ 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
// 	{ 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
// 	{ 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
// 	{ 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
// 	{ 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
// 	{ 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
// 	{ 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
// 	{ 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
// 	{ 80, 183, 189 }, { 128, 128, 0 }
// };

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
	{ 255, 56, 56 }, { 255, 157, 151 }, { 255, 112, 31 },
	{ 255, 178, 29 }, { 207, 210, 49 }, { 72, 249, 10 },
	{ 146, 204, 23 }, { 61, 219, 134 }, { 26, 147, 52 },
	{ 0, 212, 187 }, { 44, 153, 168 }, { 0, 194, 255 },
	{ 52, 69, 147 }, { 100, 115, 255 }, { 0, 24, 236 },
	{ 132, 56, 255 }, { 82, 0, 133 }, { 203, 56, 255 },
	{ 255, 149, 200 }, { 255, 55, 199 }
};

// ./yolov8-seg ../../../../weights/rs19/FP16/optimized_fp16_224_224.engine  ../../../../data/rs00058.jpg ../../../../output/
int main(int argc, char** argv)
{
	// cuda:0
	cudaSetDevice(0);

	const std::string engine_file_path{ argv[1] };
	const std::string path{ argv[2] };
	const std::string save_path{ argv[3] };

	// 输出保存路径 ... 
	string::size_type iPos = engine_file_path.find_last_of('/') + 1;
	string filename = engine_file_path.substr(iPos, engine_file_path.length() - iPos);
	string fname = filename.substr(0, filename.rfind("."));
	// std::cout << fname << std::endl;
	string f_save = save_path + fname + ".png";

	std::vector<std::string> imagePathList;
	bool isVideo{ false };

	assert(argc == 4);

	auto yolov8 = new YOLOv8_seg(engine_file_path);
	yolov8->make_pipe(true);

	if (IsFile(path))
	{
		std::string suffix = path.substr(path.find_last_of('.') + 1);
		if (
			suffix == "jpg" ||
				suffix == "jpeg" ||
				suffix == "png"
			)
		{
			imagePathList.push_back(path);
		}
		else if (
			suffix == "mp4" ||
				suffix == "avi" ||
				suffix == "m4v" ||
				suffix == "mpeg" ||
				suffix == "mov" ||
				suffix == "mkv"
			)
		{
			isVideo = true;
		}
		else
		{
			printf("suffix %s is wrong !!!\n", suffix.c_str());
			std::abort();
		}
	}
	else if (IsFolder(path))
	{
		cv::glob(path + "/*.jpg", imagePathList);
	}
	
	cv::Mat res, image;
	int topk = 100;
	int seg_channels = 32;
	float score_thres = 0.25f;
	float iou_thres = 0.65f;
	int loop_count = 20;

	// int seg_h = 56; 
	// int seg_w = 56; 
	// cv::Size size = cv::Size{ 224, 224 };

	// int seg_h = 120; 
	// int seg_w = 120; 
	// cv::Size size = cv::Size{ 480, 480 };

	int seg_h = 160; //  proto height， size d4结果
	int seg_w = 160; //  proto width
	cv::Size size = cv::Size{ 640, 640 };

	std::vector<Object> objs;

	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	if (isVideo)
	{
		cv::VideoCapture cap(path);

		if (!cap.isOpened())
		{
			printf("can not open %s\n", path.c_str());
			return -1;
		}
		while (cap.read(image))
		{
			objs.clear();
			yolov8->copy_from_Mat(image, size);
			auto start = std::chrono::system_clock::now();
			yolov8->infer();
			auto end = std::chrono::system_clock::now();
			yolov8->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
			yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
			auto tc = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
			printf("cost %2.4lf ms\n", tc);
			cv::imshow("result", res);
			if (cv::waitKey(10) == 'q')
			{
				break;
			}
		}
	}
	else
	{
		for (auto& path : imagePathList)
		{
			objs.clear();

			auto t0 = std::chrono::system_clock::now();
			image = cv::imread(path); // 10.4690 ms imread,  0.5650 ms pre
			auto t11 = std::chrono::system_clock::now();
			yolov8->copy_from_Mat(image, size);
			auto t1 = std::chrono::system_clock::now();

			// 模型推理
			auto t2 = std::chrono::system_clock::now();
			for (int i = 0; i < loop_count; ++i)
				yolov8->infer();
			auto t3 = std::chrono::system_clock::now();


			// auto start = std::chrono::system_clock::now();
			// yolov8->infer();
			// auto end = std::chrono::system_clock::now();
			yolov8->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
			auto t4 = std::chrono::system_clock::now();

			yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);

			// auto tc = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;


			auto tc = (double) std::chrono::duration_cast<std::chrono::microseconds>(t11 - t0).count() / 1000. ;
			auto tc0 = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t11).count() / 1000. ;
			auto tc1 = (double) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000. / loop_count;
			auto tc2 = (double) std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000. ;
			auto total = tc0 + tc1 + tc2;

			// printf("cost %2.4lf ms\n", tc);
			printf("%2d images cost: %2.4lf ms imread,  %2.4lf ms pre, %2.4lf ms inference,  %2.4lf ms post, %2.4lf ms total \n", loop_count, tc, tc0, tc1, tc2, total);

			// cv::imshow("result", res);
			cv::imwrite(f_save, res);
			// cv::waitKey(0);
		}
	}
	cv::destroyAllWindows();
	delete yolov8;
	return 0;
}
