#include <iostream>
#include <cstdio>
#include <Kinect.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <math.h>
#include <time.h>
#include <Windows.h>

using	namespace	std;
using	namespace	cv;

clock_t start, stop;  //clock_t是clock()函数返回类型，定义两个变量
double t;
const	int	OK_MOUSE = 5;		//鼠标开始移动的阈值，越大越稳定，越小越精确
const   float OK_Distance = 0.12;
const	int	HAND_UP = 100;		//手掌上方可能存在指尖的区域，单位毫米
const	int	HAND_LEFT_RIGHT = 100;	//手掌左右方可能存在指尖的区域，单位毫米
Vec3b	COLOR_TABLE[] = { Vec3b(255,0,0),Vec3b(0,255,0),Vec3b(0,0,255),Vec3b(255,255,255),Vec3b(0,0,0) };
enum { BLUE, GREEN, RED, WHITE, BLACK };
void	draw_window(Mat &, const DepthSpacePoint &, DepthSpacePoint &);
bool	depth_rage_check(int, int, int, int);
bool	distance_check(const CameraSpacePoint &, const CameraSpacePoint &);
bool	check_new_point(DepthSpacePoint & front, DepthSpacePoint & now, int height, int width);
float   count_PDM(Point p_0, Point p_1, Point p_3, float radius);
float   count_cosAngle(Point p_0, Point p_1, Point p_3);
int	main(void)
{

	//gm->SetSocket();

	IKinectSensor	* mySensor = nullptr;
	GetDefaultKinectSensor(&mySensor);
	mySensor->Open();

	IFrameDescription	* myDescription = nullptr;

	int	depthHeight = 0, depthWidth = 0;
	IDepthFrameSource	* myDepthSource = nullptr;
	IDepthFrameReader	* myDepthReader = nullptr;
	IDepthFrame		* myDepthFrame = nullptr;
	mySensor->get_DepthFrameSource(&myDepthSource);
	myDepthSource->get_FrameDescription(&myDescription);
	myDescription->get_Height(&depthHeight);
	myDescription->get_Width(&depthWidth);
	myDepthSource->OpenReader(&myDepthReader);			//以上为Depth帧的准备，直接开好Reader


	IBodyIndexFrameSource	* myBodyIndexSource = nullptr;
	IBodyIndexFrameReader	* myBodyIndexReader = nullptr;
	IBodyIndexFrame		* myBodyIndexFrame = nullptr;
	mySensor->get_BodyIndexFrameSource(&myBodyIndexSource);
	myBodyIndexSource->OpenReader(&myBodyIndexReader);		//以上为BodyIndex帧的准备,直接开好Reader


	IBodyFrameSource	* myBodySource = nullptr;
	IBodyFrameReader	* myBodyReader = nullptr;
	IBodyFrame		* myBodyFrame = nullptr;
	mySensor->get_BodyFrameSource(&myBodySource);
	myBodySource->OpenReader(&myBodyReader);			//以上为Body帧的准备，直接开好Reader

	ICoordinateMapper	* myMapper = nullptr;
	mySensor->get_CoordinateMapper(&myMapper);			//Maper的准备

	DepthSpacePoint		front = { 0,0 };				//用来记录上一次鼠标的位置
	DepthSpacePoint		depthUpLeft = { 0,0 };          //操作窗口的左上角和右下角，要注意这两个X和X、Y和Y的差会作为除数，所以不能都为0
	DepthSpacePoint		depthDownRight = { 0,0 };

	depthUpLeft.Y = 100;
	depthUpLeft.X = (depthWidth / 2) - 50;

	depthDownRight.Y = depthHeight / 2 + 100;
	depthDownRight.X = depthWidth - 50;


	Mat original_depth_16U(depthHeight, depthWidth, CV_16UC1);    //建立图像矩阵
	Mat copy_original_depth_16U(depthHeight, depthWidth, CV_16UC1);    //建立图像矩阵
	Mat m_middepth8u(424, 512, CV_8UC1);

	int	windowWidth = 0;	//计算操作窗口的尺寸
	int	windowHeight = 0;

	BOOLEAN left_down = false;
	BOOLEAN right_down = false;
	BOOLEAN middle_down = false;

	while (1)
	{
		start = clock();//开始记录被测函数运行前的时刻，不在测试范围的变量写在测试之前
		Mat	img(depthHeight, depthWidth, CV_8UC3);

		while (myDepthReader->AcquireLatestFrame(&myDepthFrame) != S_OK);		//读取Depth数据
		UINT	depthBufferSize = 0;
		UINT16	* depthBuffer = nullptr;;
		myDepthFrame->AccessUnderlyingBuffer(&depthBufferSize, &depthBuffer);

		myDepthFrame->CopyFrameDataToArray(depthHeight * depthWidth, (UINT16 *)original_depth_16U.data); //先把数据存入16位的图像矩阵中

																										 //SaveDepthImg(original_depth_16U);
		original_depth_16U.copyTo(copy_original_depth_16U);
		// 首先处理深度为 0 的点，这种实际上无法测量的点，
		//所以将深度为 0 的点设成最大深度
		for (int i = 0; i < 424; i++)
		{
			for (int j = 0; j < 512; j++)
			{
				m_middepth8u.at<unsigned char>(i, j) = 0;
				unsigned short & temp = copy_original_depth_16U.at<unsigned short>(i, j);
				if (temp == 0)
				{
					temp = 65535;//16位最大值为65535
				}
			}
		}

		////高斯平滑滤波
		GaussianBlur(copy_original_depth_16U, copy_original_depth_16U, Size(9, 9), 0.85, 0.85);

		//找到滤波后最小深度和最大深度
		double minValue, maxValue;
		minMaxIdx(copy_original_depth_16U, &minValue, &maxValue);


		while (myBodyIndexReader->AcquireLatestFrame(&myBodyIndexFrame) != S_OK);	//读取BodyIndex数据
		UINT	bodyIndexBufferSize = 0;
		BYTE	* bodyIndexBuffer = nullptr;
		myBodyIndexFrame->AccessUnderlyingBuffer(&bodyIndexBufferSize, &bodyIndexBuffer);


		while (myBodyReader->AcquireLatestFrame(&myBodyFrame) != S_OK);			//读取Body数据
		int	bodyBufferSize = 0;
		myBodySource->get_BodyCount(&bodyBufferSize);
		IBody	** bodyArray = new IBody *[bodyBufferSize];
		for (int i = 0; i < bodyBufferSize; i++)
			bodyArray[i] = nullptr;
		myBodyFrame->GetAndRefreshBodyData(bodyBufferSize, bodyArray);

		
		for (int i = 0; i < bodyBufferSize; i++)					//遍历6个人
		{
			BOOLEAN		result = false;
			if (bodyArray[i]->get_IsTracked(&result) == S_OK && result)
			{
				_Joint	jointArray[JointType_Count];				//将关节点输出，正式开始处理
				bodyArray[i]->GetJoints(JointType_Count, jointArray);


				windowWidth = (int)depthDownRight.X - (int)depthUpLeft.X;	//计算操作窗口的尺寸
				windowHeight = (int)depthDownRight.Y - (int)depthUpLeft.Y;

				draw_window(img, depthUpLeft, depthDownRight);				//画出操作窗口



				if (jointArray[JointType_HandRight].TrackingState == TrackingState_Tracked)
				{
					CameraSpacePoint	cameraHandRight = jointArray[JointType_HandRight].Position;
					DepthSpacePoint		depthHandRight;
					myMapper->MapCameraPointToDepthSpace(cameraHandRight, &depthHandRight);

					for (int i = depthHandRight.Y + HAND_UP; i > depthHandRight.Y - HAND_UP; i--)
						for (int j = depthHandRight.X - HAND_LEFT_RIGHT; j < depthHandRight.X + HAND_LEFT_RIGHT; j++)
						{
							if (!depth_rage_check(j, i, depthWidth, depthHeight))					//判断坐标是否合法
								continue;

							int	index = i * depthWidth + j;
							CameraSpacePoint	cameraTemp;
							DepthSpacePoint		depthTemp = { j,i };

							myMapper->MapDepthPointToCameraSpace(depthTemp, depthBuffer[index], &cameraTemp);

							if (bodyIndexBuffer[index] <= 5 && distance_check(cameraHandRight, cameraTemp) && (original_depth_16U.at<unsigned short>(i, j) < minValue + 200))
							{
								img.at<Vec3b>(i, j) = COLOR_TABLE[WHITE];
								m_middepth8u.at<unsigned char>(i, j) = 255;
							}


						}
				}

				break;
			}


		}

		

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		//寻找轮廓    
		findContours(m_middepth8u, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_NONE);
		

		// 找到最大的轮廓    

		
		int index = -1;
		double area, maxArea(0);
		for (int i = 0; i < contours.size(); i++)
		{
			area = contourArea(Mat(contours[i]));
			if (area > maxArea)
			{
				maxArea = area;
				index = i;
			}
		}

		if (index >= 0)
		{

			vector<Point> couPoint = contours[index];
			drawContours(m_middepth8u, contours, index, Scalar(0, 0, 255), 2, 2, hierarchy);

			Moments moment = moments(m_middepth8u, true);
			Point center(moment.m10 / moment.m00, moment.m01 / moment.m00);

			//circle(m_middepth8u, center, 8, Scalar(0, 0, 255), CV_FILLED);

			float radius = 0;
			Point inscribed_center = (0, 0);

			for(int i = center.y + 5;i >= center.y - 5;i--)
				for (int j = center.x - 5; j <= center.x + 5; j++)
				{
					float mindistance = 100000;
					if (!depth_rage_check(j, i, depthWidth, depthHeight))					//判断坐标是否合法
						continue;
					for (int k = 0; k < couPoint.size(); k++)
					{
						float distance = sqrt(pow(i - couPoint[k].y, 2) + pow(j - couPoint[k].x, 2));
						if (distance < mindistance)
						{
							mindistance = distance;
						}
					}

					if (mindistance > radius)
					{
						inscribed_center.y = i;
						inscribed_center.x = j;
						radius = mindistance;
					}

				}


			circle(m_middepth8u, inscribed_center, radius, Scalar(0, 0, 0), 0);
			

			float Max = 0;
			float Min = 100000;
			float size = couPoint.size();
			vector<Point> polygonPoint;
			vector<float> PDM_save;
			int count = 0;
			if (size > 10)
			{
				while (1)
				{

					Max = 0;
					Min = 100000;
					int min_index = 0;

					float PDM1 = count_PDM(couPoint[couPoint.size() - 1], couPoint[0], couPoint[1], radius);
					PDM_save.push_back(PDM1);
					if (PDM1 > Max)
					{
						Max = PDM1;
					}
					if (PDM1 < Min)
					{
						Min = PDM1;
						min_index = 0;
					}

					for (int i = 1; i < couPoint.size()-1; i++)
					{
						float PDM = count_PDM(couPoint[i - 1], couPoint[i], couPoint[i + 1], radius);
						PDM_save.push_back(PDM);
						if (PDM > Max)
						{
							Max = PDM;
						}
						if (PDM < Min)
						{
							Min = PDM;
							min_index = i;
						}
					}

					float PDM3 = count_PDM(couPoint[couPoint.size() - 2], couPoint[couPoint.size() - 1], couPoint[0], radius);
					PDM_save.push_back(PDM3);
					if (PDM3 > Max)
					{
						Max = PDM3;
					}
					if (PDM3 < Min)
					{
						Min = PDM3;
						min_index = couPoint.size() - 1;
					}

					if ((Min > 0.6))
					{
						cout << "count is:  " << count << endl;
						count = 0;

						break;
					}

					//cout << "max is:  " << Max << "  Min is:  " << Min << endl;
					
					
					for (int j = 0; j < PDM_save.size(); j++)
					{
						if (PDM_save[j] == Min)
						{
							;
						}
						else
						{
							polygonPoint.push_back(couPoint[j]);
						}
					}

					/*for (int j = 1; j < couPoint.size() - 1; j++)
					{

						if (j == min_index)
						{
							;
						}
						else
						{
							Point p = couPoint[j];
							polygonPoint.push_back(p);
						}
					}*/


					couPoint.clear();
					/*for (int i = 0; i < polygonPoint.size(); i++)
					{
						couPoint.push_back(polygonPoint[i]);
					}*/
					couPoint.swap(polygonPoint);
					polygonPoint.clear();
					PDM_save.clear();
					count++;
					
				}
			}

			vector<float> polygonPointcosAngle;
			vector<float> polygonPointdistance; 
			vector<float> polygonPointCrossProducts;
			if (couPoint.size() > 0)
			{
				int	index_center = inscribed_center.y * depthWidth + inscribed_center.x;
				CameraSpacePoint	cameraTemp_center;
				DepthSpacePoint		depthTemp_center = { inscribed_center.x,inscribed_center.y };

				myMapper->MapDepthPointToCameraSpace(depthTemp_center, depthBuffer[index_center], &cameraTemp_center);

				cout << couPoint.size() << endl;
				for (int i = 0; i < couPoint.size(); i++)
				{
					int	index = couPoint[i].y * depthWidth + couPoint[i].x;
					CameraSpacePoint	cameraTemp;
					DepthSpacePoint		depthTemp = { couPoint[i].x,couPoint[i].y };

					myMapper->MapDepthPointToCameraSpace(depthTemp, depthBuffer[index], &cameraTemp);

					float distance = sqrt(pow(cameraTemp.X - cameraTemp_center.X, 2) + pow(cameraTemp.Y - cameraTemp_center.Y, 2)+ pow(cameraTemp.Z - cameraTemp_center.Z, 2));
					polygonPointdistance.push_back(distance);

					if (i == couPoint.size() - 1)
					{
						line(img, couPoint[i], couPoint[0], COLOR_TABLE[RED], 1, 8, 0);
						float cosAngle = count_cosAngle(couPoint[i - 1], couPoint[i], couPoint[0]);
						float cross_product = (couPoint[i - 1].x - couPoint[i].x) * (couPoint[0].y - couPoint[i].y) - (couPoint[0].x - couPoint[i].x) * (couPoint[i - 1].y - couPoint[i].y);
						polygonPointcosAngle.push_back(cosAngle);
						polygonPointCrossProducts.push_back(cross_product);
					}
					else
					{
						
						line(img, couPoint[i], couPoint[i + 1], COLOR_TABLE[RED], 1, 8, 0);
						if (i == 0)
						{
							float cosAngle = count_cosAngle(couPoint[couPoint.size() - 1], couPoint[i], couPoint[i+1]);
							float cross_product = (couPoint[couPoint.size() - 1].x - couPoint[i].x) * (couPoint[i+1].y - couPoint[i].y) - (couPoint[i+1].x - couPoint[i].x) * (couPoint[couPoint.size() - 1].y - couPoint[i].y);
							polygonPointcosAngle.push_back(cosAngle);
							polygonPointCrossProducts.push_back(cross_product);
						}
						else
						{
							float cosAngle = count_cosAngle(couPoint[i - 1], couPoint[i], couPoint[i + 1]);
							float cross_product = (couPoint[i - 1].x - couPoint[i].x) * (couPoint[i + 1].y - couPoint[i].y) - (couPoint[i + 1].x - couPoint[i].x) * (couPoint[i - 1].y - couPoint[i].y);
							polygonPointcosAngle.push_back(cosAngle);
							polygonPointCrossProducts.push_back(cross_product);
						}
					}
				}
			}


			if (couPoint.size()> 0)
			{
				for (int i = 0; i < couPoint.size(); i++)
				{
					//cout << "第" << i << "个点的距离为： " << polygonPointdistance[i] << endl;
					if ((polygonPointcosAngle[i] > 0.5) && (polygonPointdistance[i] > 0.07)&&(polygonPointCrossProducts[i]>0))
					{
						circle(img, couPoint[i], 3, COLOR_TABLE[GREEN], FILLED);
					}
				}

			}

			stop = clock();//记录被测函数运行完的时刻
			t = (double)(stop - start) / CLK_TCK;//时刻差除以常量CLK_TCK(计算机程序每秒打的点数，不同计算机不一样)
			cout << "time is :" << t << endl;
			
		}

		imshow("TEST", img);
		imshow("m_middepth8u", m_middepth8u);
		if (waitKey(30) == VK_ESCAPE)
			break;

		myDepthFrame->Release();
		myBodyIndexFrame->Release();
		myBodyFrame->Release();
		delete[] bodyArray;
	}

	myBodySource->Release();
	myBodyIndexSource->Release();
	myDepthSource->Release();
	myBodyReader->Release();
	myBodyIndexReader->Release();
	myDepthReader->Release();
	myDescription->Release();
	myMapper->Release();
	mySensor->Close();
	mySensor->Release();

	return	0;
}


void	draw_window(Mat & img, const DepthSpacePoint & UpLeft, DepthSpacePoint & DownRight)
{
	Point	a = { (int)UpLeft.X,(int)DownRight.Y };
	circle(img, a, 5, COLOR_TABLE[RED], -1, 0, 0);
	Point	b = { (int)UpLeft.X,(int)UpLeft.Y };
	circle(img, b, 5, COLOR_TABLE[GREEN], -1, 0, 0);
	Point	c = { (int)DownRight.X,(int)UpLeft.Y };
	circle(img, c, 5, COLOR_TABLE[BLUE], -1, 0, 0);
	Point	d = { (int)DownRight.X,(int)DownRight.Y };
	circle(img, d, 5, COLOR_TABLE[WHITE], -1, 0, 0);
	line(img, a, b, COLOR_TABLE[RED], 1, 8, 0);
	line(img, b, c, COLOR_TABLE[RED], 1, 8, 0);
	line(img, c, d, COLOR_TABLE[RED], 1, 8, 0);
	line(img, a, d, COLOR_TABLE[RED], 1, 8, 0);
}
bool	depth_rage_check(int x, int y, int depthWidth, int depthHeight)
{
	if (x >= 0 && x < depthWidth && y >= 0 && y < depthHeight)
		return	true;
	return	false;
}

bool	distance_check(const CameraSpacePoint & hand, const CameraSpacePoint & temp)
{
	float a = sqrt(pow(hand.X - temp.X, 2) + pow(hand.Y - temp.Y, 2) + pow(hand.Z - temp.Z, 2));
	//cout << a << endl;
	if (a <= OK_Distance)
		return	true;
	return	false;
}
bool	check_new_point(DepthSpacePoint & front, DepthSpacePoint & now, int height, int width)
{
	if (now.X == width - 1 && now.Y == height - 1 && (front.X || front.Y))
		return	false;
	else	if (fabs(now.X - front.X) <= OK_MOUSE && fabs(now.Y - front.Y) <= OK_MOUSE)
		return	false;
	return	true;
}

float count_PDM(Point p_0, Point p_1, Point p_3, float radius)
{
	//method from :https://www.jstage.jst.go.jp/article/transinf/E96.D/3/E96.D_750/_pdf
	float a = sqrt(pow(p_1.x - p_3.x, 2) + pow(p_1.y - p_3.y, 2));
	float b = sqrt(pow(p_1.x - p_0.x, 2) + pow(p_1.y - p_0.y, 2));
	float c = sqrt(pow(p_3.x - p_0.x, 2) + pow(p_3.y - p_0.y, 2));

	float sin_half = sqrt((1 - (c*c - a*a - b*b) / (2 * a*b)) / 2);

	float PDM = 2*a*b*sin_half / ((a + b)*radius);
	return PDM;
}

float count_cosAngle(Point p_0, Point p_1, Point p_3)
{
	float a = sqrt(pow(p_1.x - p_3.x, 2) + pow(p_1.y - p_3.y, 2));
	float b = sqrt(pow(p_1.x - p_0.x, 2) + pow(p_1.y - p_0.y, 2));
	float c = sqrt(pow(p_3.x - p_0.x, 2) + pow(p_3.y - p_0.y, 2));

	float cosAngle = (a*a + b*b - c*c) / (2 * a*b);
	return cosAngle;
}




