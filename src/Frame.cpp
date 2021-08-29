#include "Frame.h"
#include "Settings.h"
#include <algorithm>
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "CycleTimer.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Sequence.h"
// For fast edge detection using structure forests
#include <opencv2/ximgproc.hpp>

using namespace cv;
namespace EdgeVO{

Frame::Frame()
    :m_image() , m_depthMap()
{}

Frame::Frame(Mat& image)
    :m_image(image) , m_depthMap( Mat() )
{}

Frame::Frame(std::string imagePath, std::string depthPath, Sequence* seq)
    :m_image(cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE)) , m_depthMap(cv::imread(depthPath, cv::ImreadModes::IMREAD_UNCHANGED)) , 
    m_imageName(imagePath), m_depthName(depthPath), m_seq(seq)
{
#ifdef DISPLAY_LOGS 
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;

    std::cout << "imagePath: " << imagePath << std::endl;
    std::cout << "depthPath: " << depthPath << std::endl;
    std::cout << "m_image: " << m_image.cols << " x " << m_image.rows << "(ch: " << m_image.channels() << ")" << std::endl;    
    std::cout << "m_depthMap: " << m_depthMap.cols << " x " << m_depthMap.rows << "(ch: " << m_depthMap.channels() << ")" << std::endl;
#endif

    //m_image.convertTo(m_image, CV_32FC1);
    //m_depthMap.convertTo(m_depthMap, CV_32FC1, EdgeVO::Settings::PIXEL_TO_METER_SCALE_FACTOR);
    m_pyramidImageUINT.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidImage.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidDepth.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidEdge.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidLaplacian.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Ddx.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Ddy.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Idx.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Idy.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_gradMag.resize(EdgeVO::Settings::PYRAMID_DEPTH); // for ADVO
    m_pyramid_Gdx.resize(EdgeVO::Settings::PYRAMID_DEPTH); // for ADVO
    m_pyramid_Gdy.resize(EdgeVO::Settings::PYRAMID_DEPTH); // for ADVO

    m_pyramidImageUINT[0] = m_image.clone(); 
    m_pyramidImage[0] = m_image;
    m_pyramidImage[0].convertTo(m_pyramidImage[0], CV_32FC1);
    m_pyramidDepth[0] = m_depthMap;    
    m_pyramidDepth[0].convertTo(m_pyramidDepth[0], CV_32FC1, EdgeVO::Settings::PIXEL_TO_METER_SCALE_FACTOR);
    //m_pyramidImage.push_back(m_image);
#ifdef SFORESTS_EDGES
    m_sforestDetector = m_seq->getSFDetector();//cv::ximgproc::createStructuredEdgeDetection("../model/SForestModel.yml");
    m_pyramidImageSF.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidImageSF[0] = cv::imread(m_imageName,cv::ImreadModes::IMREAD_COLOR);
#else
    m_sforestDetector = nullptr;
#endif
    
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif
}

Frame::Frame(Mat& image, Mat& depthMap)
    :m_image(image) , m_depthMap(depthMap)
{}

/*
Frame::Frame(const Frame& cp)
: m_image( cp.m_image ) , m_depthMap( cp.m_depthMap )
{}
*/
Frame::~Frame()
{
    releaseAllVectors();
}

void Frame::releaseAllVectors()
{
    m_pyramidImage.clear();
    m_pyramid_Idx.clear();
    m_pyramid_Idy.clear();
    m_pyramid_gradMag.clear();
    m_pyramid_Gdx.clear();
    m_pyramid_Gdy.clear();
    m_pyramidDepth.clear();
    m_pyramidMask.clear();
    m_pyramidEdge.clear();
    m_pyramidLaplacian.clear();
    m_pyramidImageUINT.clear();
    //m_pyramidImageFloat.clear();
}

int Frame::getHeight(int lvl) const
{
    return m_pyramidImage[lvl].rows;
}
int Frame::getWidth(int lvl) const
{
    return m_pyramidImage[lvl].cols;
}
void Frame::printPaths() 
{
    std::cout << m_imageName << std::endl;
    std::cout << m_depthName << std::endl;
}
/*
Frame& Frame::operator=(const Frame& rhs)
{
    if(this == &rhs)
        return *this;
    // copy and swap
    Frame temp(rhs);
    std::swap(*this, temp);
    return *this;
}
*/
Mat& Frame::getImageForDisplayOnly()
{
    return m_pyramidImageUINT[0];
}
Mat& Frame::getEdgeForDisplayOnly()
{
    return m_pyramidEdge[0];
}
Mat& Frame::getDepthForDisplayOnly()
{
    return m_pyramidDepth[0];
}

cv::Mat Frame::getGdx(int lvl) const
{
    return (m_pyramid_Gdx[lvl].clone()).reshape(1, m_pyramid_Gdx[lvl].rows * m_pyramid_Gdx[lvl].cols);
}
cv::Mat Frame::getGdy(int lvl) const
{
    return (m_pyramid_Gdy[lvl].clone()).reshape(1, m_pyramid_Gdy[lvl].rows * m_pyramid_Gdy[lvl].cols);
}

Mat Frame::getImage(int lvl) const
{
    return m_pyramidImage[lvl].clone();
}
cv::Mat Frame::getImageVector(int lvl) const
{
    return (m_pyramidImage[lvl].clone()).reshape(1, m_pyramidImage[lvl].rows * m_pyramidImage[lvl].cols);
}
Mat Frame::getGradientMagVector(int lvl) const
{
    return (m_pyramid_gradMag[lvl].clone()).reshape(1, m_pyramid_gradMag[lvl].rows * m_pyramid_gradMag[lvl].cols);
}

Mat Frame::getDepthMap(int lvl) const
{   
    return (m_pyramidDepth[lvl].clone()).reshape(1, m_pyramidDepth[lvl].rows * m_pyramidDepth[lvl].cols);
}

Mat Frame::getMask(int lvl) const
{
    return m_pyramidMask[lvl].clone();
}
Mat Frame::getEdges(int lvl) const
{
    return (m_pyramidEdge[lvl].clone()).reshape(1, m_pyramidEdge[lvl].rows * m_pyramidEdge[lvl].cols);
}
cv::Mat Frame::getLaplacian(int lvl) const
{
    return (m_pyramidLaplacian[lvl].clone()).reshape(1, m_pyramidLaplacian[lvl].rows * m_pyramidLaplacian[lvl].cols);
}
cv::Mat Frame::getDepthGradientX(int lvl) const
{
    return (m_pyramid_Ddx[lvl].clone()).reshape(1, m_pyramid_Ddx[lvl].rows * m_pyramid_Ddx[lvl].cols);
}
cv::Mat Frame::getDepthGradientY(int lvl) const
{
    return (m_pyramid_Ddy[lvl].clone()).reshape(1, m_pyramid_Ddy[lvl].rows * m_pyramid_Ddy[lvl].cols);
}
cv::Mat Frame::getGradientX(int lvl) const
{
    return (m_pyramid_Idx[lvl].clone()).reshape(1, m_pyramid_Idx[lvl].rows * m_pyramid_Idx[lvl].cols);
}
cv::Mat Frame::getGradientY(int lvl) const
{
    return (m_pyramid_Idy[lvl].clone()).reshape(1, m_pyramid_Idy[lvl].rows * m_pyramid_Idy[lvl].cols);
}


void Frame::makePyramids(bool flagMasf)
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif

    #if ADAPTIVE_DVO_FULL
    createPyramid(m_pyramidImage[0], m_pyramidImage, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_LINEAR, flagMasf);
    #else
    createPyramid(m_pyramidImage[0], m_pyramidImage, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_LINEAR);
    #endif
    cv::buildPyramid(m_pyramidImageUINT[0], m_pyramidImageUINT, EdgeVO::Settings::PYRAMID_BUILD);

    #if EDGEVO_FULL
    createPyramid(m_pyramidDepth[0], m_pyramidDepth, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    #else
    createPyramid(m_pyramidDepth[0], m_pyramidDepth, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    //createPyramidWithBilateralFiltering(m_pyramidDepth[0], m_pyramidDepth, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_NEAREST);    
    #endif

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before CANNY" << std::endl;
#endif

#ifdef CANNY_EDGES
    // Canny
    createCannyEdgePyramids();
#elif LoG_EDGES
    // LoG
    createLoGEdgePyramids();
#elif SFORESTS_EDGES
    createStructuredForestEdgePyramid();
#elif CONV_BASIN
    createBasinPyramids();
#else
    // Sobel
    createSobelEdgePyramids();
#endif

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before ADVO" << std::endl;
#endif

#if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE | ADAPTIVE_DVO_WITHOUT_GRAD
    createImageGradientPyramids(true);
    createDepthGradientPyramids();
#else
    createImageGradientPyramids();
#endif    

    ///////////////////////////////////////////////
    // To Do list at 2021.04.11.
    // create pyramids of X3D and normals.
    // Too expensive... Change point-to-plane error based on the depth loss of Kerl13Iros.
    ///////////////////////////////////////////////

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif
}

void Frame::createPyramid(cv::Mat& src, std::vector<cv::Mat>& dst, int pyramidSize, int interpolationFlag, bool flagMASP)
{    
    dst.resize(pyramidSize);        
    dst[0] = src;

    for(size_t i = 1; i < pyramidSize; ++i){            
        cv::resize(dst[i-1], dst[i],cv::Size(0, 0), 0.5, 0.5, interpolationFlag);

        if(flagMASP && i == pyramidSize - 1){
            edgeSmoothingFilter(dst[i]);
        }         
    }
}

void Frame::createPyramidWithBilateralFiltering(cv::Mat& src, std::vector<cv::Mat>& dst, int pyramidSize, int interpolationFlag)
{
    cv::Mat img_tmp;
    float sigma_s = 8.0;
    float sigma_d = 0.2;

    dst.resize(pyramidSize);
    img_tmp = src.clone(); cv::bilateralFilter(img_tmp, dst[0], 5, sigma_d, sigma_s);
    dst[0] = src;
    for(size_t i = 1; i < pyramidSize; ++i){        
        cv::resize(dst[i-1], dst[i],cv::Size(0, 0), 0.5, 0.5, interpolationFlag);        
        img_tmp = dst[i].clone(); cv::bilateralFilter(img_tmp, dst[i], 5, sigma_d, sigma_s);
    }
}

void Frame::createDepthGradientPyramids()
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif

    int one(1);
    int zero(0);
    double scale = 0.5;

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before calcGradientX" << std::endl;
#endif

    // Dx
    calcGradientX(m_pyramidDepth[0], m_pyramid_Ddx[0]);
    // Dy
    calcGradientY(m_pyramidDepth[0], m_pyramid_Ddy[0]);

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before createPyramid" << std::endl;
#endif

    createPyramid(m_pyramid_Ddx[0], m_pyramid_Ddx, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_NEAREST);
    createPyramid(m_pyramid_Ddy[0], m_pyramid_Ddy, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_NEAREST); 

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif   
}

void Frame::createImageGradientPyramids(bool flagLaplacian)
{
    int one(1);
    int zero(0);
    double scale = 0.5;

    // Ix
    calcGradientX(m_pyramidImage[0], m_pyramid_Idx[0]);
    // Iy
    calcGradientY(m_pyramidImage[0], m_pyramid_Idy[0]);
   
    createPyramid(m_pyramid_Idx[0], m_pyramid_Idx, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    createPyramid(m_pyramid_Idy[0], m_pyramid_Idy, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);

    if(flagLaplacian){
        // gradient magnitude pyramid.
        //m_pyramid_gradMag[0] = cv::Mat::zeros(m_pyramid_Idx[0].size(), CV_32FC1);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Is it correct?
        // m_pyramid_gradMag[0] = m_pyramid_Idx[0].mul(m_pyramid_Idx[0]) + m_pyramid_Idy[0].mul(m_pyramid_Idy[0]);
        cv::sqrt(m_pyramid_Idx[0].mul(m_pyramid_Idx[0]) + m_pyramid_Idy[0].mul(m_pyramid_Idy[0]), m_pyramid_gradMag[0]);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        createPyramid(m_pyramid_gradMag[0], m_pyramid_gradMag, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
        // Ixx, Iyy
        calcGradientX(m_pyramid_gradMag[0], m_pyramid_Gdx[0]);
        createPyramid(m_pyramid_Gdx[0], m_pyramid_Gdx, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
        calcGradientY(m_pyramid_gradMag[0], m_pyramid_Gdy[0]);
        createPyramid(m_pyramid_Gdy[0], m_pyramid_Gdy, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
        // Laplacian
        //m_pyramidLaplacian[0] = m_pyramid_Idxx[0] + m_pyramid_Idyy[0];
        calcLaplacian(m_pyramidImage[0], m_pyramidLaplacian[0]);
        createPyramid(m_pyramidLaplacian[0], m_pyramidLaplacian, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    }
}

void Frame::calcGradientX(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f).clone();
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(x == 0)
                dst.at<float>(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x));
            else if(x == src.cols-1)
                dst.at<float>(y,x) = (src.at<float>(y,x) - src.at<float>(y,x-1));
            else
                dst.at<float>(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x-1))*0.5;
        }
    }
}
void Frame::calcGradientY(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f ).clone();
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(y == 0)
                dst.at<float>(y,x) = (src.at<float>(y+1,x) - src.at<float>(y,x));
            else if(y == src.rows-1)
                dst.at<float>(y,x) = (src.at<float>(y,x) - src.at<float>(y-1,x));
            else
                dst.at<float>(y,x) = (src.at<float>(y+1,x) - src.at<float>(y-1,x))*0.5;
        }
    }
}
void Frame::calcLaplacian(cv::Mat& src, cv::Mat& dst)
{
    int ww = src.cols, hh = src.rows;
    dst = cv::Mat(hh, ww, CV_32FC1, 0.f ).clone();
    float tLap;
    for(int j=1; j<hh-1; j++){
		for(int i=1; i<ww-1; i++){

			// BASIC method.
			tLap = 8*src.at<float>(j,i);
			for(int m=-1; m<=1; m++){
				for(int n=-1; n<=1; n++){
					if(m == 0 && n == 0) continue;
					tLap -= src.at<float>(j+m,i+n);
				}
			}
            dst.at<float>(j,i) = tLap/8.0f;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////
// Smooth intensity pixels where the gradient magnitude is large only.
void Frame::edgeSmoothingFilter(cv::Mat &in_img)
/////////////////////////////////////////////////////////////////////////////
{
    cv::Mat img_grad_x, img_grad_y, img_grad_mag;
    cv::Mat tmp_img = in_img.clone();

    // Ix
    calcGradientX(in_img, img_grad_x);
    // Iy
    calcGradientY(in_img, img_grad_y);
   
    cv::sqrt(img_grad_x.mul(img_grad_x) + img_grad_y.mul(img_grad_y), img_grad_mag);

    float *p_img_grad_mag = img_grad_mag.ptr<float>();
    unsigned char *p_in_img = in_img.ptr<unsigned char>();
    unsigned char *p_tmp_img = tmp_img.ptr<unsigned char>();

    int ww = in_img.cols, hh = in_img.rows;
    
    // Smooth pixel intensity where gradient magnitude is higher than threshold.
    for(int j=1; j<hh-1; ++j){
        for(int i=1; i<ww-1; ++i){
            
            float avg = 0.0f;
            int idx = j*ww + i;

            ////////////////////////////////////////////////
            if(p_img_grad_mag[idx] >= EdgeVO::Settings::EDGE_SMOOTH_GRADIENT_THRESH){
            ////////////////////////////////////////////////
                for(int n=-1; n<=1; ++n){
                    for(int k=-1; k<=1; ++k){
                        int idx_win = idx + n*ww + k;
                        avg += (float)p_in_img[idx_win];
                    }
                }
                avg /= 9.0f;
                p_tmp_img[idx] = (unsigned char)avg;
            }
        }
    }

    in_img = tmp_img.clone();

}

void Frame::createCannyEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_thresh; //not used
        //cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
  /// Canny detector
        float upperThreshold = cv::threshold(m_pyramidImageUINT[i], img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
        Canny(m_pyramidImageUINT[i], m_pyramidEdge[i], lowerThresh, upperThreshold, 3, true);
    }
    //void Canny(InputArray image, OutputArray edges, float threshold1, float threshold2, int apertureSize=3, bool L2gradient=false )
}

void Frame::createLoGEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_dest; 
        cv::GaussianBlur( m_pyramidImageUINT[i], img_dest, Size(3,3), 0, 0, cv::BORDER_DEFAULT );
        cv::Laplacian( img_dest, img_dest, CV_8UC1, 3, 1., 0, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( img_dest, img_dest );
        cv::threshold(img_dest, m_pyramidEdge[i], 25, 255, cv::THRESH_BINARY);
    }

}
void Frame::createSobelEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        cv::Mat grad_x, grad_y;
        cv::Mat grad;
        /// x Gradient
        Sobel( m_pyramidImageUINT[i], grad_x, CV_16S, 1, 0, 3, 1., 0, cv::BORDER_DEFAULT );
        convertScaleAbs( grad_x, grad_x );
        /// y Gradient
        Sobel( m_pyramidImageUINT[i], grad_y, CV_16S, 0, 1, 3, 1., 0, cv::BORDER_DEFAULT );
        convertScaleAbs( grad_y, grad_y );
        addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
        double max;
        double min;
        cv::minMaxLoc(grad, &min, &max);
        cv::threshold(grad/max, m_pyramidEdge[i], 0.95, 255, cv::THRESH_BINARY);
    }

}
void Frame::createStructuredForestEdgePyramid()
{
    cv::buildPyramid(m_pyramidImageSF[0], m_pyramidImageSF, EdgeVO::Settings::PYRAMID_BUILD);
    for(size_t i = 0; i < m_pyramidImageSF.size(); ++i)
    {
        Mat image = m_pyramidImageSF[i].clone();
        image.convertTo(image, CV_32FC3, 1./255.0);
        cv::Mat edges(image.size(), image.type());
        m_sforestDetector->detectEdges(image, edges );
        cv::threshold(edges, m_pyramidEdge[i], 0.15, 255, cv::THRESH_BINARY);
     
    }
        

}
void Frame::createBasinPyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_thresh; //not used
        cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
  /// Canny detector
        float upperThreshold = cv::threshold(m_pyramidImageUINT[i], img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
        Canny(m_pyramidImageUINT[i], m_pyramidEdge[i], lowerThresh, upperThreshold, 3, true);
    }
    m_pyramidEdge[m_pyramidEdge.size()-1] = cv::Mat::ones(m_pyramidImageUINT[m_pyramidEdge.size()-1].rows, m_pyramidImageUINT[m_pyramidEdge.size()-1].cols, CV_8UC1);
}

bool Frame::hasDepthMap()
{
    return !(m_depthMap.empty() );

}

void Frame::setDepthMap(Mat& depthMap)
{
    if(!hasDepthMap())
        m_depthMap = depthMap;
    // Otherwise do nothing
}

}