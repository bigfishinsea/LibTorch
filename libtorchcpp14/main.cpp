#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <bits/stdc++.h>

using namespace std;

vector<double> matrixTToEuler(vector<vector<double>>& T);

class Detector {
public:
	/***
	 * @brief constructor
	 * @param model_path - path of the TorchScript weight file
	 * @param device_type - inference with CPU/GPU
	 */
	Detector(const std::string& model_path, bool use_gpu);

private:
	torch::jit::script::Module module_;
	torch::Device device_;
	bool half_;
};

Detector::Detector(const std::string& model_path, bool use_gpu) :device_(torch::kCPU)
{
	if (torch::cuda::is_available() && use_gpu)
	{
		//std::cout << "use cuda...\n";
		device_ = torch::kCUDA;
	}
	else
	{
		//std::cout << "use cpu...\n";
		device_ = torch::kCPU;
	}

	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module_ = torch::jit::load(model_path);
	}
	catch (const c10::Error& e) {
		std::cerr << "Error loading the model!\n";
		std::exit(EXIT_FAILURE);
	}

	half_ = (device_ != torch::kCPU);
	module_.to(device_);

	if (half_) {
		module_.to(torch::kHalf);
	}

	module_.eval();
}


//int main()
//{
//	std::shared_ptr<Detector> detector = std::make_shared<Detector>("C:/Users/dell/Desktop/111/neural_network_singlecom_test.pt", false);
//	return 0;
//}

int maintest()
{
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Predicting on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Predicting on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	//调试好后注释掉
	//device_type = torch::kCPU;
	torch::Device device(device_type);

	//Init model
	std::string model_pb = "C:/Users/dell/Desktop/111/neural_network_singlecom_test.pt";
	auto module = torch::jit::load(model_pb, device);
	module.to(at::kCUDA);

	int maxTurn = 100;
	int nowTurn = 1;

	while (nowTurn <= maxTurn) {
		// Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		vector<float> inputvec = { -1.04719, 0.34907, 1.57080, 0, -1.91986, 0 };
		inputs.push_back(torch::from_blob(inputvec.data(), 6, torch::kFloat).to(at::kCUDA));


		// Execute the model and turn its output into a tensor.
		auto o = module.forward(inputs);
		at::Tensor output = o.toTensor();
		cout << output << endl;

		// Tensor张量转vector数组
		// 要转到CPU上，不转就报错
		at::Tensor t = output.toType(torch::kFloat).clone().to(at::kCPU);
		cout << t << endl;
		cout << "vector数组元素如下：" << endl;
		std::vector<float> outputvec(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
		for (int i = 0; i < outputvec.size(); i++) {
			cout << outputvec[i] << endl;
		}
		nowTurn++;
	}

	return 0;
}


int main_4gan()
{
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Predicting on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Predicting on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	//调试好后注释掉
	//device_type = torch::kCPU;
	torch::Device device(device_type);

	//Init model
	std::string model_pb = "C:/Users/dell/Desktop/simulatefloder/loopEuler/dataset_1113/b3_network_30_5.pt";
	auto module = torch::jit::load(model_pb, device);
	module.to(at::kCUDA);

	int maxTurn = 1000;
	int nowTurn = 1;

	long long time1 = 0;
	long long time2 = 0;

vector<double> prismatics = {
0,
-0.00249714,
-0.009966621,
-0.022343715,
-0.03952761,
-0.061383229,
-0.087748813,
-0.118445782,
-0.153281369,
-0.192055313,
-0.234570108,
-0.280627137,
-0.330034038,
-0.38260467,
-0.43815826,
-0.496520039,
-0.557522557,
-0.621003864,
-0.68680809,
-0.754784664,
-0.824787365,
-0.896674371,
-0.970306731,
-1.045548684,
-1.122266705,
-1.200328785,
-1.279605707,
-1.35996806,
-1.441289328,
-1.523442683,
-1.606302638,
-1.689744743,
-1.773643906,
-1.857875928,
-1.942316734,
-2.026841695,
-2.1113271,
-2.195649412,
-2.279684278,
-2.363307413,
-2.446394123,
-2.528819525,
-2.610458176,
-2.691183926,
-2.770869856,
-2.849388076,
-2.926609482,
-3.002402352,
-3.07663434,
-3.1491706,
-3.219873832,
-3.288600576,
-3.355206787,
-3.419544526,
-3.481461365,
-3.540794549,
-3.597377733,
-3.651043876,
-3.701618467,
-3.74890825,
-3.792726755,
-3.832884873,
-3.86917865,
-3.901406772,
-3.929381137,
-3.952917788,
-3.971844428,
-3.986015837,
-3.995313876,
-3.999657441,
-3.999009414,
-3.993376805,
-3.982808308,
-3.967392245,
-3.947252333,
-3.922541296,
-3.893433674,
-3.860117513,
-3.822788725,
-3.781646372,
-3.736888132,
-3.6887088,
-3.637296795,
-3.582836994,
-3.525505544,
-3.465472227,
-3.402900119,
-3.337947105,
-3.270764881,
-3.201501519,
-3.130301663,
-3.057303474,
-2.98264606,
-2.906463017,
-2.828887659,
-2.750051508,
-2.670081994,
-2.589107781,
-2.507255055 };

vector<double> prismaticv = {
0,
- 0.499083067,
- 0.993815851,
- 1.480127043,
- 1.954470541,
- 2.413884653,
- 2.856131477,
- 3.279704264,
- 3.683694344,
- 4.067722997,
- 4.431865559,
- 4.776455319,
- 5.102007552,
- 5.409151446,
- 5.698549685,
- 5.970855949,
- 6.226716424,
- 6.466723804,
- 6.691422646,
- 6.901291958,
- 7.096759147,
- 7.278182991,
- 7.445877942,
- 7.600104453,
- 7.7410819,
- 7.869001334,
- 7.984003632,
- 8.086232856,
- 8.175771722,
- 8.252715566,
- 8.317134417,
- 8.369067245,
- 8.408552588,
- 8.435616585,
- 8.450276248,
- 8.452537243,
- 8.442401245,
- 8.419863976,
- 8.384914004,
- 8.337528702,
- 8.277666225,
- 8.205286953,
- 8.120333759,
- 8.022728495,
- 7.912377523,
- 7.78916759,
- 7.652960466,
- 7.503561866,
- 7.340775439,
- 7.164360273,
- 6.974037731,
- 6.769400627,
- 6.550072699,
- 6.315628821,
- 6.065591233,
- 5.799311889,
- 5.51614986,
- 5.215547818,
- 4.896898569,
- 4.559432741,
- 4.202646455,
- 3.826178944,
- 3.429860758,
- 3.013876001,
- 2.578820065,
- 2.125897368,
- 1.657200105,
- 1.175139265,
- 0.683038637,
- 0.185017733,
0.314517122,
0.811249433,
1.301034034,
1.780102644,
2.245246545,
2.693952148,
3.124448035,
3.535650626,
3.927023422,
4.298501942,
4.650297527,
4.982856525,
5.296747616,
5.59263369,
5.871190365,
6.133078653,
6.378915744,
6.609260765,
6.824605167,
7.025389233,
7.2119917,
7.384729736,
7.543910743,
7.689742112,
7.822440404,
7.942193237,
8.049103866,
8.143315676,
8.224920105
};

nowTurn = 0;
maxTurn = prismatics.size();


	while (nowTurn < maxTurn) {
		// Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		vector<float> inputvec = { (float)prismatics[nowTurn], (float)prismaticv[nowTurn] };
		inputs.push_back(torch::from_blob(inputvec.data(), 2, torch::kFloat).to(at::kCUDA));


		// Execute the model and turn its output into a tensor.
		auto o = module.forward(inputs);
		at::Tensor output = o.toTensor();
		// cout << output << endl;

		// Tensor张量转vector数组
		// 要转到CPU上，不转就报错
		at::Tensor t = output.toType(torch::kFloat).clone().to(at::kCPU);
		// cout << t << endl;
		//cout << "vector数组元素如下：" << endl;
		std::vector<float> outputvec(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
		cout << outputvec[5] << "\n";
		//for (int i = 0; i < outputvec.size(); i++) {
		//	cout << outputvec[i] << "\n";
		//}
		//if (nowTurn == 500) {
		//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//}
		//if (nowTurn == 1000) {
		//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//}
		nowTurn++;
		//cout << "\n";
	}

	//std::cout << "平均耗时（毫秒）:" << (time2 - time1) / 500 << std::endl;

	return 0;
}

// 计算四杆机构（新）
int main() {
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Predicting on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Predicting on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	//调试好后注释掉
	//device_type = torch::kCPU;
	torch::Device device(device_type);

	//Init model
	std::string model_pb = "C:/Users/dell/Desktop/simulatefloder/loopEuler/dataset_1113/b3_network_30_5.pt";
	auto module = torch::jit::load(model_pb, device);
	module.to(at::kCUDA);

	std::ifstream csv_data("C:/Users/dell/Desktop/simulatefloder/loopEuler/Train_1113/Result.csv", std::ios::in);
	std::string line;

	if (!csv_data.is_open())
	{
		std::cout << "Error: opening file fail" << std::endl;
		std::exit(1);
	}

	std::istringstream sin;         //将整行字符串line读入到字符串istringstream中
	std::string word;

	vector<vector<float>> Fills;

	string input1 = "\"prismatic.s\"";
	string input2 = "\"prismatic.v\"";

	map<int, int> IdxToFill;

	// 读取标题行
	std::getline(csv_data, line);
	sin.clear();
	sin.str(line);
	int nowidx = 0;
	while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
	{
		if (word == input1) {
			IdxToFill[nowidx] = 0;
		}
		else if (word == input2) {
			IdxToFill[nowidx] = 1;
		}
		nowidx++;
	}

	// 读取数据
	while (std::getline(csv_data, line))
	{
		sin.clear();
		sin.str(line);
		int nowidx = 0;
		vector<float> nowfill(2, 0.0);
		while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
		{
			if (IdxToFill.count(nowidx)) {
				nowfill[IdxToFill[nowidx]] = atof(word.c_str());
			}
			nowidx++;
		}
		Fills.push_back(nowfill);
	}
	csv_data.close();


	int nowTurn = 0;
	int maxTurn = Fills.size();


	while (nowTurn < maxTurn) {
		std::vector<torch::jit::IValue> inputs;
		vector<float> inputvec = { Fills[nowTurn][0], Fills[nowTurn][1] };
		inputs.push_back(torch::from_blob(inputvec.data(), 2, torch::kFloat).to(at::kCUDA));


		// Execute the model and turn its output into a tensor.
		auto o = module.forward(inputs);
		at::Tensor output = o.toTensor();
		// cout << output << endl;

		// Tensor张量转vector数组
		// 要转到CPU上，不转就报错
		at::Tensor t = output.toType(torch::kFloat).clone().to(at::kCPU);
		std::vector<float> outputvec(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
		cout << outputvec[5] << "\n";
		nowTurn++;
	}

	return 0;
}


int main_robot6()
{
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Predicting on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Predicting on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	//调试好后注释掉
	//device_type = torch::kCPU;
	torch::Device device(device_type);

	//Init model
	std::string model_pb = "C:/Users/dell/Desktop/simulatefloder/robotnn1222/dataset/neural_network_b61_90_10000.pt";
	auto module = torch::jit::load(model_pb, device);
	module.to(at::kCUDA);

	std::ifstream csv_data("C:/Users/dell/Desktop/simulatefloder/robotnn0426/Train/Result.csv", std::ios::in);
	std::string line;

	if (!csv_data.is_open())
	{
		std::cout << "Error: opening file fail" << std::endl;
		std::exit(1);
	}

	std::istringstream sin;         //将整行字符串line读入到字符串istringstream中
	std::string word;

	vector<vector<float>> Fills;

	string input1 = "\"revolute_fit1.phi\"";
	string input2 = "\"revolute_fit2.phi\"";
	string input3 = "\"revolute_fit3.phi\"";
	string input4 = "\"revolute_fit4.phi\"";
	string input5 = "\"revolute_fit5.phi\"";
	string input6 = "\"revolute_fit6.phi\"";

	map<int, int> IdxToFill;

	// 读取标题行
	std::getline(csv_data, line);
	sin.clear();
	sin.str(line);
	int nowidx = 0;
	while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
	{
		if (word == input1) {
			IdxToFill[nowidx] = 0;
		}
		else if (word == input2) {
			IdxToFill[nowidx] = 1;
		}
		else if (word == input3) {
			IdxToFill[nowidx] = 2;
		}
		else if (word == input4) {
			IdxToFill[nowidx] = 3;
		}
		else if (word == input5) {
			IdxToFill[nowidx] = 4;
		}
		else if (word == input6) {
			IdxToFill[nowidx] = 5;
		}
		nowidx++;
	}

	// 读取数据
	while (std::getline(csv_data, line))
	{
		sin.clear();
		sin.str(line);
		int nowidx = 0;
		vector<float> nowfill(6, 0.0);
		while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
		{
			if (IdxToFill.count(nowidx)) {
				nowfill[IdxToFill[nowidx]] = atof(word.c_str());
			}
			nowidx++;
		}
		Fills.push_back(nowfill);
	}
	csv_data.close();
	

	int nowTurn = 0;
	int maxTurn = Fills.size();


	while (nowTurn < maxTurn) {
		// Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		vector<float> inputvec = { Fills[nowTurn][0], Fills[nowTurn][1], Fills[nowTurn][2], Fills[nowTurn][3], Fills[nowTurn][4], Fills[nowTurn][5] };
		//for (int i = 0; i < inputvec.size(); i++) {
		//	cout << inputvec[i] << " ";
		//}
		//cout << endl;
		inputs.push_back(torch::from_blob(inputvec.data(), 6, torch::kFloat).to(at::kCUDA));


		// Execute the model and turn its output into a tensor.
		auto o = module.forward(inputs);
		at::Tensor output = o.toTensor();
		// cout << output << endl;

		// Tensor张量转vector数组
		// 要转到CPU上，不转就报错
		at::Tensor t = output.toType(torch::kFloat).clone().to(at::kCPU);
		std::vector<float> outputvec(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
		cout << outputvec[5] << "\n";
		nowTurn++;
	}

	//std::cout << "平均耗时（毫秒）:" << (time2 - time1) / 500 << std::endl;

	return 0;
}


int computeEulerFromFile()
{
	std::ifstream csv_data("C:/Users/dell/Desktop/simulatefloder/robotnn0426/Result_matrix.csv", std::ios::in);
	std::string line;

	if (!csv_data.is_open())
	{
		std::cout << "Error: opening file fail" << std::endl;
		std::exit(1);
	}

	std::istringstream sin;         //将整行字符串line读入到字符串istringstream中
	std::string word;

	vector<vector<vector<double>>> Fills;

	string input1 = "b61.frame_a.R.T[11]";
	string input2 = "b61.frame_a.R.T[12]";
	string input3 = "b61.frame_a.R.T[13]";
	string input4 = "b61.frame_a.R.T[21]";
	string input5 = "b61.frame_a.R.T[22]";
	string input6 = "b61.frame_a.R.T[23]";
	string input7 = "b61.frame_a.R.T[31]";
	string input8 = "b61.frame_a.R.T[32]";
	string input9 = "b61.frame_a.R.T[33]";

	map<int, int> IdxToFill;

	// 读取标题行
	std::getline(csv_data, line);
	sin.clear();
	sin.str(line);
	int nowidx = 0;
	while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
	{
		if (word == input1) {
			IdxToFill[nowidx] = 0;
		}
		else if (word == input2) {
			IdxToFill[nowidx] = 1;
		}
		else if (word == input3) {
			IdxToFill[nowidx] = 2;
		}
		else if (word == input4) {
			IdxToFill[nowidx] = 3;
		}
		else if (word == input5) {
			IdxToFill[nowidx] = 4;
		}
		else if (word == input6) {
			IdxToFill[nowidx] = 5;
		}
		else if (word == input7) {
			IdxToFill[nowidx] = 6;
		}
		else if (word == input8) {
			IdxToFill[nowidx] = 7;
		}
		else if (word == input9) {
			IdxToFill[nowidx] = 8;
		}
		nowidx++;
	}

	// 读取数据
	while (std::getline(csv_data, line))
	{
		sin.clear();
		sin.str(line);
		int nowidx = 0;
		vector<vector<double>> nowfill(4, vector<double>(4, 0.0));
		while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
		{
			if (IdxToFill.count(nowidx)) {
				int idx = IdxToFill[nowidx];
				int row = (idx / 3) + 1;
				int col = (idx % 3) + 1;
				nowfill[row][col] = stod(word.c_str());
			}
			nowidx++;
		}
		Fills.push_back(nowfill);
	}
	csv_data.close();


	for (int i = 0; i < 3; i++) {
		int nowTurn = 0;
		int maxTurn = Fills.size();

		while (nowTurn < maxTurn) {
			vector<double> nowans = matrixTToEuler(Fills[nowTurn]);
			cout << nowans[i] << "\n";
			nowTurn++;
		}
		cout << "-------------------------------\n";
	}

	return 0;
}


vector<double> matrixTToEuler(vector<vector<double>>& T) {
	vector<double> euler(3, 0.0);
	euler[0] = atan2(-1 * T[3][1], sqrt(T[1][1] * T[1][1] + T[2][1] * T[2][1]));
	euler[1] = atan2(T[2][1] / cos(euler[0]), T[1][1] / cos(euler[0]));
	euler[2] = atan2(T[3][2] / cos(euler[0]), T[3][3] / cos(euler[0]));
	return euler;
}