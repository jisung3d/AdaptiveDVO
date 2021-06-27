#ifndef EDGEDIRECTVO_H
#define EDGEDIRECTVO_H

#include <iostream>
#include "Sequence.h"
#include "Trajectory.h"
#include "Statistics.h"
#include "Pose.h"
#include <Eigen/Core>
#include <fstream>

#define SQUARE(x) ((x) * (x))

namespace EdgeVO{

class EdgeDirectVO{
    public:
        EdgeDirectVO();
        ~EdgeDirectVO();
        EdgeDirectVO& operator=(const EdgeDirectVO& rhs);
        EdgeDirectVO(const EdgeDirectVO& cp);

        // Main Algorithm //
        void runEdgeDirectVO();
        void solveSystemOfEquations(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate);
        //void runAdaptiveDirectVO();
        void solveSystemOfEquationsForADVO(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate);
        // Helper Algorithms //
        void prepareVectors(int lvl);
        void warpAndCalculateResiduals(const Pose& pose, const std::vector<float>& Z, const std::vector<bool>& E, const int h, const int w, const cv::Mat& cameraMatrix, const int lvl);

        void outputPose(const Pose& pose, double timestamp);

        void prepare3DPoints( );
        void make3DPoints(const cv::Mat& cameraMatrix, int lvl);
        float warpAndProject(const Eigen::Matrix<double,4,4>& invPose, int lvl, bool flagGradMax = false);
        float warpAndProjectForAdaptiveDVO(const Eigen::Matrix<double,4,4>& invPose, int lvl);
        float interpolateVector(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>& toInterp, float x, float y, int w) const;
        bool checkBounds(float x, float xlim, float y, float ylim, float oldZ, float newZ, bool edgePixel);

        void terminationRequested();

    private:
        
        // Convenience classes
        Sequence m_sequence;
        Trajectory m_trajectory;
        Statistics m_statistics;

        std::ofstream m_outputFile;

        // Image Vectors and residual vectors
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im1;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im2;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im1Final;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im2Final;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_residual;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_residual_D;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_rsquared;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_rsquared_D;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_grad1;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_grad2;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_grad1Final;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_grad2Final;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_residual_grad;  // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_rsquared_grad;  // for ADVO
        // Matrices and vectors for solving Least Squares problem
        // poseupdate = -((1+lambda).*(w.*J))\(w.*r);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_weights;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_weights_D;
        Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> m_Jacobian;        
        Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> m_Jacobian_D;  // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_weights_grad;   // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> m_Jacobian_grad;   // for ADVO
        


        // Depth of Image 1
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_Z;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_D2;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_D1Final;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_D2Final;
        // Warped x,y image coordinates
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_warpedX;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_warpedY;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_warpedZ;

        // Vector of 3D points and Transformed 3D points
        std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> m_X3DVector;
        Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> m_X3D;
        Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> m_newX3D;        
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_XFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_YFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_ZFinal;

        // Vectors of Depth Gradients
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gxD;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gyD;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gxDFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gyDFinal;

        // Vectors of Image Gradients
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gx;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gy;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gxFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gyFinal;
        
        // Vectors of Image Laplacians
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_L;  // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_Gx;  // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_Gy;  // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_GxFinal;  // for ADVO
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_GyFinal;  // for ADVO
        
        // Mask for edge pixels as well as to prevent out of bounds, NaN, etc.
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::RowMajor> m_finalMask;
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::RowMajor> m_finalMaskGrad;  // for ADVO
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::RowMajor> m_edgeMask;
        float m_lambda;

        //Currently unused
        Eigen::Matrix<float, 6 , 6, Eigen::RowMajor> m_JTJ;
        Eigen::Matrix<float, 6 , Eigen::RowMajor> m_JTr;
        Eigen::Matrix<double, 6 , Eigen::RowMajor> m_poseupdate;
        Eigen::Matrix<float, 6 , 6, Eigen::RowMajor> m_A;
        std::vector<float> m_x;
        std::vector<float> m_y;

        //////////////////////////////////////////////////////////////////////////
        // For Adaptive DVO
        //////////////////////////////////////////////////////////////////////////
        // for RGB-D test setting.
	    const float m_th_grad_sq[4] ={SQUARE(1),SQUARE(2),SQUARE(3),SQUARE(4)};
	    // const float m_th_grad_2_sq[4] ={SQUARE(20),SQUARE(30),SQUARE(30),SQUARE(30)};
        //const float m_th_grad_sq[4] ={SQUARE(2),SQUARE(5),SQUARE(10),SQUARE(15)};
        //const float m_th_grad_sq[4] ={5, 10, 15, 20};
	    const float m_th_grad_2_sq[4] ={SQUARE(20),SQUARE(30),SQUARE(30),SQUARE(30)};
	    const float m_th_D_sq[4] ={SQUARE(0.02f),SQUARE(0.03f),SQUARE(0.04f),SQUARE(0.05f)};	

        // int m_cnt_gradient = 0;
        // int m_cnt_glr = 0;
        // int m_cnt_l_sq = 0;
	    //////////////////////////////////////////////////////////////////////////
};

}
#endif //EDGEDIRECTVO_H