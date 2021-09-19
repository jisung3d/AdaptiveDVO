#include <iostream>
#include "EdgeDirectVO.h"
#include "CycleTimer.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pose.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdio.h> 
#include <stdlib.h> 
#include <random>
#include <iterator>
#include <algorithm>

#include <Settings.h>

//#define DISPLAY_SEQUENCE

namespace EdgeVO{
    using namespace cv;
EdgeDirectVO::EdgeDirectVO()
    :m_sequence(EdgeVO::Settings::ASSOC_FILE) , m_trajectory() , 
     m_lambda(0.), m_avg_disp_prev(0.0f)
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif
    int length = m_sequence.getFrameHeight( getBottomPyramidLevel() ) * m_sequence.getFrameWidth( getBottomPyramidLevel() );
    m_X3DVector.resize(EdgeVO::Settings::PYRAMID_DEPTH); // Vector for each pyramid level
    for(size_t i = 0; i < m_X3DVector.size(); ++i){
        m_X3DVector[i].resize(length / std::pow(4, i) , Eigen::NoChange); //3 Vector for each pyramid for each image pixel
    }

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before m_X3D" << std::endl;
#endif

    m_X3D.resize(length, Eigen::NoChange);
    m_warpedX.resize(length);
    m_warpedY.resize(length);
    m_warpedZ.resize(length);
    m_gxD.resize(length);
    m_gxDFinal.resize(length);
    m_gyD.resize(length);
    m_gyDFinal.resize(length);
    m_gx.resize(length);
    m_gxFinal.resize(length);
    m_gy.resize(length);
    m_gyFinal.resize(length);
    m_im1.resize(length);
    m_im2.resize(length);
    m_im1Final.resize(length);
    m_im2Final.resize(length);
    m_ZFinal.resize(length);

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before m_Z" << std::endl;
#endif

    m_Z.resize(length);    
    m_D2.resize(length);    
    m_D1Final.resize(length);
    m_D2Final.resize(length);

    m_edgeMask.resize(length);
    
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before m_L" << std::endl;
#endif

    m_L.resize(length);  // for ADVO
    m_Gx.resize(length);  // for ADVO
    m_Gy.resize(length);  // for ADVO
    m_GxFinal.resize(length);  // for ADVO
    m_GyFinal.resize(length); 

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before m_outputFile" << std::endl;
#endif

    m_outputFile.open(EdgeVO::Settings::RESULTS_FILE);    
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif
}

EdgeDirectVO::EdgeDirectVO(const EdgeDirectVO& cp)
    :m_sequence(EdgeVO::Settings::ASSOC_FILE)
{}

EdgeDirectVO::~EdgeDirectVO()
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif
    m_outputFile.close();

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif
}

EdgeDirectVO& EdgeDirectVO::operator=(const EdgeDirectVO& rhs)
{
    if(this == &rhs)
        return *this;
    
    EdgeDirectVO temp(rhs);
    std::swap(*this, temp);
    return *this;

}

// void EdgeDirectVO::runAdaptiveDirectVO()
// {
// #ifdef DISPLAY_LOGS
//     std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
// #endif

//     //Start timer for stats
//     m_statistics.start();

//     //Make Pyramid for Reference frame
//     m_sequence.makeReferenceFramePyramids();

//     // Run for entire sequence
//     //Prepare some vectors
//     prepare3DPoints();

//     //Init camera_pose with ground truth trajectory to make comparison easy
//     Pose camera_pose = m_trajectory.initializePoseToGroundTruth(m_sequence.getFirstTimeStamp());
//     Pose keyframe_pose = camera_pose;
//     // relative_pose intiialized to identity matrix
//     Pose relative_pose;

//     // Start clock timer
//     outputPose(camera_pose, m_sequence.getFirstTimeStamp());
//     m_statistics.addStartTime((float) EdgeVO::CycleTimer::currentSeconds());

//     for (size_t n = 0; m_sequence.sequenceNotFinished(); ++n)
//     {
//         std::cout << std::endl << camera_pose << std::endl;

// #ifdef DISPLAY_SEQUENCE
//         //We re-use current frame for reference frame info
//         m_sequence.makeCurrentFramePyramids();

//         //Display images
//         int keyPressed1 = m_sequence.displayCurrentImage();
//         int keyPressed2 = m_sequence.displayCurrentEdge();
//         int keyPressed3 = m_sequence.displayCurrentDepth();
//         if(keyPressed1 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY 
//             || keyPressed2 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY
//             || keyPressed3 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY) 
//         {
//             terminationRequested();
//             break;
//         }
//         //Start algorithm timer for each iteration
//         float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
// #else
//         //Start algorithm timer for each iteration
//         float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
//         m_sequence.makeCurrentFramePyramids();
// #endif //DISPLAY_SEQUENCE

//         if( n % EdgeVO::Settings::KEYFRAME_INTERVAL == 0 )
//         {
//             keyframe_pose = camera_pose;
//             relative_pose.setIdentityPose();
//         }

//         //Constant motion assumption
// #ifdef DISPLAY_LOGS
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - updateKeyFramePose" << std::endl;
// #endif
//         relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.getLastRelativePose());
// #ifdef DISPLAY_LOGS
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - setPose" << std::endl;
// #endif
//         relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));

//         //Constant acc. assumption
//         //relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.get2LastRelativePose());
//         //relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));
        
//         // For each image pyramid level, starting at the top, going down
//         for (int lvl = getTopPyramidLevel(); lvl >= getBottomPyramidLevel(); --lvl)
//         {
// #ifdef DISPLAY_LOGS
//             std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - prepareVectors" << std::endl;
// #endif
//             const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
//             prepareVectors(lvl);
            
//             //make3DPoints(cameraMatrix, lvl);            

//             float lambda = 0.f;
//             float error_last = EdgeVO::Settings::INF_F;
//             float error = error_last;
//             for(int i = 0; i < EdgeVO::Settings::MAX_ITERATIONS_PER_PYRAMID[ lvl ]; ++i)
//             {
// #ifdef DISPLAY_LOGS
//                 std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - warpAndProject" << std::endl;
// #endif
//                 error_last = error;
//                 error = warpAndProject(relative_pose.inversePoseEigen(), lvl);

//                 ///////////////////////////////////////////////////////////////////////////
//                 // This part should be changed to test another Loss functions.
//                 ///////////////////////////////////////////////////////////////////////////
//                 // Levenberg-Marquardt
//                 if( error < error_last)
//                 {
//                     // Update relative pose
//                     Eigen::Matrix<double, 6 , Eigen::RowMajor> del;
//                     solveSystemOfEquations(lambda, lvl, del);
//                     //std::cout << del << std::endl;
                    
//                     if( (del.segment<3>(0)).dot(del.segment<3>(0)) < EdgeVO::Settings::MIN_TRANSLATION_UPDATE & 
//                         (del.segment<3>(3)).dot(del.segment<3>(3)) < EdgeVO::Settings::MIN_ROTATION_UPDATE    )
//                         break;

//                     cv::Mat delMat = se3ExpEigen(del);
//                     relative_pose.updatePose( delMat );

//                     //Update lambda
//                     if(lambda <= EdgeVO::Settings::LAMBDA_MAX)
//                         lambda = EdgeVO::Settings::LAMBDA_MIN;
//                     else
//                         lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
//                 }
//                 else
//                 {
//                     if(lambda == EdgeVO::Settings::LAMBDA_MIN)
//                         lambda = EdgeVO::Settings::LAMBDA_MAX;
//                     else
//                         lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
//                 }
//                 ///////////////////////////////////////////////////////////////////////////
//             }
//         }

// #ifdef DISPLAY_LOGS
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - updateKeyFramePose" << std::endl;
// #endif
//         camera_pose.updateKeyFramePose(keyframe_pose.getPoseMatrix(), relative_pose.getPoseMatrix());
// #ifdef DISPLAY_LOGS
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - outputPose" << std::endl;
// #endif
//         outputPose(camera_pose, m_sequence.getCurrentTimeStamp());
//         //At end, update sequence for next image pair
//         float endTime = (float) EdgeVO::CycleTimer::currentSeconds();
//         m_trajectory.addPose(camera_pose);

//         // Don't time past this part (reading from disk)
// #ifdef DISPLAY_LOGS                
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - addDurationForFrame" << std::endl;
// #endif
//         m_statistics.addDurationForFrame(startTime, endTime);
//         m_statistics.addCurrentTime((float) EdgeVO::CycleTimer::currentSeconds());
// #ifdef DISPLAY_LOGS                
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - printStatistics - E" << std::endl;
// #endif
//         m_statistics.printStatistics();
// #ifdef DISPLAY_LOGS                
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - advanceSequence - E" << std::endl;
// #endif
//         if(!m_sequence.advanceSequence()) break;
// #ifdef DISPLAY_LOGS                
//         std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - advanceSequence - X" << std::endl;
// #endif
//     }
//     // End algorithm level timer
//     m_statistics.end();
// #ifdef DISPLAY_LOGS
//     std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
// #endif

//     return;
// }

void EdgeDirectVO::runEdgeDirectVO()
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif

    //Start timer for stats
    m_statistics.start();

    //Make Pyramid for Reference frame
    m_sequence.makeReferenceFramePyramids();

    // Run for entire sequence
    //Prepare some vectors
    prepare3DPoints();

    //Init camera_pose with ground truth trajectory to make comparison easy
    Pose camera_pose; // = m_trajectory.initializePoseToGroundTruth(m_sequence.getFirstTimeStamp());
    Pose keyframe_pose = camera_pose;
    // relative_pose intialized to identity matrix
    Pose relative_pose;

    // Start clock timer
    outputPose(camera_pose, m_sequence.getFirstTimeStamp());
    m_statistics.addStartTime((float) EdgeVO::CycleTimer::currentSeconds());

#ifdef DISPLAY_LOGS
    std::cout << "for (size_t n = 0; m_sequence.sequenceNotFinished(); ++n)" << std::endl;    
#endif

    for (size_t n = 0; m_sequence.sequenceNotFinished(); ++n)
    {
        std::cout << std::endl << camera_pose << std::endl;

#ifdef DISPLAY_SEQUENCE
        //We re-use current frame for reference frame info
        m_sequence.makeCurrentFramePyramids();

        //Display images
        int keyPressed1 = m_sequence.displayCurrentImage();
        int keyPressed2 = m_sequence.displayCurrentEdge();
        int keyPressed3 = m_sequence.displayCurrentDepth();
        if(keyPressed1 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY 
            || keyPressed2 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY
            || keyPressed3 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY) 
        {
            terminationRequested();
            break;
        }
        //Start algorithm timer for each iteration
        float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
#else
        //Start algorithm timer for each iteration
        float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
        // check MASF
        bool flagMasf = false;
#if MASF_FOR_LARGE_MOTION_HANDLING
        if(m_avg_disp_prev > 5.0f) flagMasf = true;
#endif
        m_sequence.makeCurrentFramePyramids(flagMasf);
    
#endif //DISPLAY_SEQUENCE

        if( n % EdgeVO::Settings::KEYFRAME_INTERVAL == 0 )
        {
            keyframe_pose = camera_pose;
            relative_pose.setIdentityPose();
        }

        //Constant motion assumption
#ifdef DISPLAY_LOGS
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - updateKeyFramePose" << std::endl;
#endif
        relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.getLastRelativePose());
#ifdef DISPLAY_LOGS
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - setPose" << std::endl;
#endif
        relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));

        //Constant acc. assumption
        //relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.get2LastRelativePose());
        //relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));
        
        // For each image pyramid level, starting at the top, going down
        for (int lvl = getTopPyramidLevel(); lvl >= getBottomPyramidLevel(); --lvl)
        {
#ifdef DISPLAY_LOGS
            std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - prepareVectors" << std::endl;
#endif
            const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
            prepareVectors(lvl);
            
            //make3DPoints(cameraMatrix, lvl);            

            float lambda = 0.f;
            float reweightDepthCost = EdgeVO::Settings::COST_RATIO;
            float error_last = EdgeVO::Settings::INF_F;
            float error = error_last;
            for(int i = 0; i < EdgeVO::Settings::MAX_ITERATIONS_PER_PYRAMID[ lvl ]; ++i)
            {
#ifdef DISPLAY_LOGS
                std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - warpAndProject" << std::endl;
#endif
                error_last = error;
#if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE | ADAPTIVE_DVO_WITHOUT_GRAD
                error = warpAndProjectForAdaptiveDVO(relative_pose.inversePoseEigen(), lvl, reweightDepthCost);
#else
                error = warpAndProject(relative_pose.inversePoseEigen(), lvl);
#endif

                ///////////////////////////////////////////////////////////////////////////
                // This part should be changed to test another Loss functions.
                ///////////////////////////////////////////////////////////////////////////
                // Levenberg-Marquardt
                if( error < error_last)
                {
                    // Update relative pose
                    Eigen::Matrix<double, 6 , Eigen::RowMajor> del;
                    #if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE | ADAPTIVE_DVO_WITHOUT_GRAD
                    solveSystemOfEquationsForADVO(lambda, lvl, del, reweightDepthCost);                    
                    #else
                    solveSystemOfEquations(lambda, lvl, del);
                    #endif
                    //std::cout << del << std::endl;
                    
                    if( (del.segment<3>(0)).dot(del.segment<3>(0)) < EdgeVO::Settings::MIN_TRANSLATION_UPDATE & 
                        (del.segment<3>(3)).dot(del.segment<3>(3)) < EdgeVO::Settings::MIN_ROTATION_UPDATE    )
                        break;

                    cv::Mat delMat = se3ExpEigen(del);
                    relative_pose.updatePose( delMat );

                    //Update lambda
                    if(lambda <= EdgeVO::Settings::LAMBDA_MAX)
                        lambda = EdgeVO::Settings::LAMBDA_MIN;
                    else
                        lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
                }
                else
                {
                    if(lambda == EdgeVO::Settings::LAMBDA_MIN)
                        lambda = EdgeVO::Settings::LAMBDA_MAX;
                    else
                        lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
                }
                ///////////////////////////////////////////////////////////////////////////
            }
        }

#ifdef DISPLAY_LOGS
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - updateKeyFramePose" << std::endl;
#endif
        camera_pose.updateKeyFramePose(keyframe_pose.getPoseMatrix(), relative_pose.getPoseMatrix());
#ifdef DISPLAY_LOGS
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - outputPose" << std::endl;
#endif
        outputPose(camera_pose, m_sequence.getCurrentTimeStamp());
        //At end, update sequence for next image pair
        float endTime = (float) EdgeVO::CycleTimer::currentSeconds();
        m_trajectory.addPose(camera_pose);

#if MASF_FOR_LARGE_MOTION_HANDLING
        //////////////////////////////////////////////////////////////////////////////
        // MASF //////////////////////////////////////////////////////////////////////
        // If the relative motion makes the average flow larger than threshold,
        // Smooth the highest layer of the next input image.
        m_avg_disp_prev = computeAverageDisparity(relative_pose.inversePoseEigen(), getTopPyramidLevel());
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - average flow magnitude : " << m_avg_disp_prev << std::endl;
        //////////////////////////////////////////////////////////////////////////////
#endif
        // Don't time past this part (reading from disk)
#ifdef DISPLAY_LOGS                
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - addDurationForFrame" << std::endl;
#endif
        m_statistics.addDurationForFrame(startTime, endTime);
        m_statistics.addCurrentTime((float) EdgeVO::CycleTimer::currentSeconds());
#ifdef DISPLAY_LOGS                
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - printStatistics - E" << std::endl;
#endif
        m_statistics.printStatistics();
#ifdef DISPLAY_LOGS                
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - advanceSequence - E" << std::endl;
#endif
        //////////////////////////////////////////////////////////////////////////
        // Update frame indices and reference frame (every 3 frames).
        if(!m_sequence.advanceSequence()) break;
        //////////////////////////////////////////////////////////////////////////
#ifdef DISPLAY_LOGS                
        std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - advanceSequence - X" << std::endl;
#endif
    }

#ifdef DISPLAY_LOGS
    std::cout << "for (size_t n = 0; m_sequence.sequenceNotFinished(); ++n) - End" << std::endl;    
#endif

    // End algorithm level timer
    m_statistics.end();
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif

    return;
}

void EdgeDirectVO::prepareVectors(int lvl)
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif
 
    cv2eigen(m_sequence.getCurrentFrame()->getEdges(lvl), m_edgeMask);
    cv2eigen(m_sequence.getReferenceFrame()->getImageVector(lvl), m_im1);
    cv2eigen(m_sequence.getCurrentFrame()->getImageVector(lvl), m_im2);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientX(lvl), m_gx);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientY(lvl), m_gy);
    cv2eigen(m_sequence.getReferenceFrame()->getDepthMap(lvl), m_Z);

#if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE | ADAPTIVE_DVO_WITHOUT_GRAD
    cv2eigen(m_sequence.getReferenceFrame()->getGradientMagVector(lvl), m_grad1);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientMagVector(lvl), m_grad2);    
    cv2eigen(m_sequence.getCurrentFrame()->getGdx(lvl), m_Gx);
    cv2eigen(m_sequence.getCurrentFrame()->getGdy(lvl), m_Gy);
    cv2eigen(m_sequence.getCurrentFrame()->getLaplacian(lvl), m_L);
    cv2eigen(m_sequence.getCurrentFrame()->getDepthMap(lvl), m_D2);
    cv2eigen(m_sequence.getCurrentFrame()->getDepthGradientX(lvl), m_gxD);
    cv2eigen(m_sequence.getCurrentFrame()->getDepthGradientY(lvl), m_gyD);
#endif
    
    size_t numElements;
////////////////////////////////////////////////////////////
// REGULAR_DIRECT_VO
////////////////////////////////////////////////////////////
#if REGULAR_DIRECT_VO
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= EdgeVO::Settings::MIN_GRADIENT_THRESH).select(0, m_edgeMask);
#elif REGULAR_DIRECT_VO_SUBSET
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
#elif ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE | ADAPTIVE_DVO_WITHOUT_GRAD                
    #if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_WITHOUT_GRAD
    // Use all pixels.
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    //m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= EdgeVO::Settings::MIN_GRADIENT_THRESH).select(0, m_edgeMask);
    m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= m_th_grad_sq[lvl]).select(0, m_edgeMask);
    //m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= 25).select(0, m_edgeMask);
    #elif ADAPTIVE_DVO_EDGE
    // Use edge pixels only.
    m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= m_th_grad_sq[lvl]).select(0, m_edgeMask);
    #endif  
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
#else
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    //m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    //size_t numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
#endif //REGULAR_DIRECT_VO

////////////////////////////////////////////////////////////
// EDGEVO_SUBSET_POINTS
////////////////////////////////////////////////////////////
#if EDGEVO_SUBSET_POINTS
    //numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
    //size_t numElements = (m_edgeMask.array() != 0).count() < EdgeVO::Settings::NUMBER_POINTS ? (m_edgeMask.array() != 0).count() : EdgeVO::Settings::NUMBER_POINTS;
    std::vector<size_t> indices, randSample;
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);

    //size_t idx = 0;
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            indices.push_back(i);
        }
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(randSample),
                numElements, std::mt19937{std::random_device{}()});
    
    //size_t idx = 0;
    for(int i = 0; i < randSample.size(); ++i)
    {
        m_im1Final[i] = m_im1[randSample[i]];
        m_ZFinal[i] = m_Z[randSample[i]];
        m_X3D.row(i) = (m_X3DVector[lvl].row(randSample[i])).array() * m_Z[randSample[i]];
        m_finalMask[i] = m_edgeMask[randSample[i]];    
    }

#elif ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE | ADAPTIVE_DVO_WITHOUT_GRAD
////////////////////////////////////////////////////////////
// Adaptive Direct VO
////////////////////////////////////////////////////////////
    numElements = (m_edgeMask.array() != 0).count();

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - numElements : " << numElements << std::endl;
#endif
    
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);

    m_grad1Final.resize(numElements);
    m_finalMaskGrad.resize(numElements);

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before masking" << std::endl;
#endif

    size_t idx = 0;    
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            m_im1Final[idx] = m_im1[i];
            m_ZFinal[idx] = m_Z[i];
            m_X3D.row(idx) = (m_X3DVector[lvl].row(i)).array() * m_Z[i];
            
            m_finalMask[idx] = m_edgeMask[i];

            //m_cnt_gradient++;

#if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE
            // check gradient-Laplarcian ratio
            float L_sq = SQUARE(m_L[i]);
            double GLR = abs(m_L[i]/m_grad1[i]);          
            
            if(L_sq > m_th_grad_2_sq[lvl]){             
                #if ADAPTIVE_DVO_WITHOUT_GLR
                m_grad1Final[idx] = m_grad1[i];
                m_finalMaskGrad[idx] = m_edgeMask[i];
                #else
                if(GLR >= EdgeVO::Settings::TH_GLR){ // reflect gradient-laplacian ratio
                    m_grad1Final[idx] = m_grad1[i];
                    m_finalMaskGrad[idx] = m_edgeMask[i];
                    //m_cnt_glr++;
                }
                else
                    m_finalMask[idx] = (unsigned char)0;
                #endif  

                //m_cnt_l_sq++;              
            }
            else{
                m_grad1Final[idx] = 0.0f;
                m_finalMaskGrad[idx] = (unsigned char)0;
            }
#endif
            ++idx;

// #ifdef DISPLAY_LOGS
//             std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - GLR_sq: " 
//                 << GLR_sq << " (th: " << m_th_grad_2_sq[lvl] << ")" << std::endl;
// #endif     
        }
    }

    //std::cout << "m_cnt_glr: " << m_cnt_glr << "m_cnt_l_sq: " << m_cnt_l_sq << ", m_cnt_gradient: " << m_cnt_gradient << std::endl;
#else //  REGULAR_DIRECT_VO true | REGULAR_DIRECT_VO_SUBSET true | EDGEVO_FULL true | ADAPTIVE_DVO_WITHOUT_GRAD true
////////////////////////////////////////////////////////////
// Edge Direct VO
////////////////////////////////////////////////////////////
    numElements = (m_edgeMask.array() != 0).count();

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - numElements" << std::endl;
#endif
    
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);
    size_t idx = 0;
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            m_im1Final[idx] = m_im1[i];
            m_ZFinal[idx] = m_Z[i];
            m_X3D.row(idx) = (m_X3DVector[lvl].row(i)).array() * m_Z[i];
            m_finalMask[idx] = m_edgeMask[i];
            ++idx;
        }
    }

#endif //EDGEVO_SUBSET_POINTS


////////////////////////////////////////////////////////////
    m_Z.resize(numElements);
    m_Z = m_ZFinal;
    m_edgeMask.resize(numElements);
    m_edgeMask = m_finalMask;

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif
}

void EdgeDirectVO::make3DPoints(const cv::Mat& cameraMatrix, int lvl)
{
    m_X3D = m_X3DVector[lvl].array() * m_Z.replicate(1, m_X3DVector[lvl].cols() ).array();
}
/////////////////////////////////////////////////////////////////////////////////////
// TO DO LIST 2021.04.06.
// Iterative Weighted Least Square (IWLS) Problem for joint optimization.
// 0. Initialize Validity Maps based on RGB and Depth gradient values.
// 1. Photoconsistency Maximization
// 2. Gradient difference minimization
// 3. Point-to-Plane distance minimization 
//  - Change point-to-plane error based on the depth loss of Kerl13Iros.
//  - r_depth = D_2(tau(x,T)) - [(TK^(-1)(x,D_1(x)))]_Z
// TO DO LIST 2021.04.11.
/////////////////////////////////////////////////////////////////////////////////////
float EdgeDirectVO::warpAndProject(const Eigen::Matrix<double,4,4>& invPose, int lvl, bool flagGradMax)
{
    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;
    //std::cout << R << std::endl << t << std::endl;
    //std::cout << "Cols: " << m_X3D[lvl].cols() << "Rows: " << m_X3D[lvl].rows() << std::endl;
    
    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() ); // transform X3D of current frame to the reference camera coordinates.

    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //std::cout << cy << std::endl;
    //exit(1);
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

    m_warpedX.resize(m_X3D.rows());
    m_warpedY.resize(m_X3D.rows());
    m_warpedZ.resize(m_X3D.rows());

    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx; // x positon of the re-projected X3D on the reference frame.
    //m_warpedX.array() += cx;
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy; // y positon of the re-projected X3D on the reference frame.    
    //m_warpedY.array() += cy;
    m_warpedZ = m_newX3D.row(2).array(); // Z value of the transformed X3D on the reference frame.

    // (R.array() < s).select(P,Q );  // (R < s ? P : Q)
    //std::cout << newX3D.rows() << std::endl;
    //std::cout << m_finalMask.rows() << std::endl;

    // Check both Z 3D points are >0
    //m_finalMask = m_edgeMask;

    // for Edge DVO
    m_finalMask = m_edgeMask;

    m_finalMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_newX3D.row(2).transpose().array() > EdgeVO::Settings::MAX_Z_DEPTH).select(0, m_finalMask);

    //m_finalMask = (m_newX3D.row(2).transpose().array() > 10.f).select(0, m_finalMask);
    m_finalMask = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_X3D.col(2).array() > 10.f).select(0, m_finalMask);
    m_finalMask = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMask, 0);
    m_finalMask = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMask, 0);
    
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMask = (m_warpedX.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array() >= w-2).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array().isFinite()).select(m_finalMask, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMask = (m_warpedY.array() >= h-2).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array().isFinite()).select(m_finalMask, 0);
    
    // for Adaptive DVO
    if(flagGradMax){
        m_finalMaskGrad = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMaskGrad);
        //m_finalMask = (m_newX3D.row(2).transpose().array() > EdgeVO::Settings::MAX_Z_DEPTH).select(0, m_finalMask);

        //m_finalMask = (m_newX3D.row(2).transpose().array() > 10.f).select(0, m_finalMask);
        m_finalMaskGrad = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMaskGrad);
        //m_finalMask = (m_X3D.col(2).array() > 10.f).select(0, m_finalMask);
        m_finalMaskGrad = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMaskGrad, 0);
        m_finalMaskGrad = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMaskGrad, 0);
        
        // Check new projected x coordinates are: 0 <= x < w-1
        m_finalMaskGrad = (m_warpedX.array() < 0.f).select(0, m_finalMaskGrad);
        m_finalMaskGrad = (m_warpedX.array() >= w-2).select(0, m_finalMaskGrad);
        m_finalMaskGrad = (m_warpedX.array().isFinite()).select(m_finalMaskGrad, 0);
        // Check new projected x coordinates are: 0 <= y < h-1
        m_finalMaskGrad = (m_warpedY.array() >= h-2).select(0, m_finalMaskGrad);
        m_finalMaskGrad = (m_warpedY.array() < 0.f).select(0, m_finalMaskGrad);
        m_finalMaskGrad = (m_warpedY.array().isFinite()).select(m_finalMaskGrad, 0);
    }
    

// If we want every point, save some computation time- see the #else
////////////////////////////////////////////////////////////
#ifdef EDGEVO_SUBSET_POINTS_EXACT
    size_t numElements = (m_finalMask.array() != 0).count() < EdgeVO::Settings::NUMBER_POINTS ? (m_finalMask.array() != 0).count() : EdgeVO::Settings::NUMBER_POINTS;

    //size_t numElements = (m_finalMask.array() != 0).count();
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    std::vector<size_t> indices, randSample;

    //size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {
            indices.push_back(i);
        }
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(randSample),
                numElements, std::mt19937{std::random_device{}()});
    
    size_t idx = 0;
    for(int i = 0; i < randSample.size(); ++i)
    {
        m_gxFinal[i]  = interpolateVector( m_gx, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_gyFinal[i]  = interpolateVector( m_gy, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_im1[i] = m_im1Final[randSample[i]];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
        m_im2Final[i] = interpolateVector(m_im2, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_XFinal[i] = m_newX3D(0,randSample[i]);
        m_YFinal[i] = m_newX3D(1,randSample[i]);
        m_ZFinal[i] = m_newX3D(2,randSample[i]);        
    }
    
////////////////////////////////////////////////////////////
#else //EDGEVO_SUBSET_POINTS_EXACT
    // For non random numbers EDGEVO_SUBSET_POINTS
    size_t numElements = (m_finalMask.array() != 0).count();
    m_gxDFinal.resize(numElements);
    m_gyDFinal.resize(numElements);
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);

    size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {
            // gradient loss            
            if(flagGradMax && m_finalMaskGrad[i] != 0){
                m_gxFinal[idx]  = interpolateVector( m_Gx, m_warpedX[i], m_warpedY[i], w);
                m_gyFinal[idx]  = interpolateVector( m_Gy, m_warpedX[i], m_warpedY[i], w);
                m_im1[idx] = m_grad1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
                m_im2Final[idx] = interpolateVector(m_grad2, m_warpedX[i], m_warpedY[i], w);
            }
            // photometric loss
            else{
                m_gxFinal[idx]  = interpolateVector( m_gx, m_warpedX[i], m_warpedY[i], w);
                m_gyFinal[idx]  = interpolateVector( m_gy, m_warpedX[i], m_warpedY[i], w);
                m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
                m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
            }
            
            m_D2Final[idx]  = interpolateVector( m_D2, m_warpedX[i], m_warpedY[i], w);
            m_gxDFinal[idx]  = interpolateVector( m_gxD, m_warpedX[i], m_warpedY[i], w);
            m_gyDFinal[idx]  = interpolateVector( m_gyD, m_warpedX[i], m_warpedY[i], w);

            m_XFinal[idx] = m_newX3D(0,i);
            m_YFinal[idx] = m_newX3D(1,i);
            m_ZFinal[idx] = m_newX3D(2,i);

            ++idx;
        }
    }
#endif //EDGEVO_SUBSET_POINTS_EXACT
////////////////////////////////////////////////////////////
    
    //apply mask to im1, im2, gx, and gy
    //interp coordinates of im2, gx, and gy
    // calc residual

    //calc A and b matrices
    // for Photometric Errors
    m_residual.resize(numElements);
    m_rsquared.resize(numElements);
    m_weights.resize(numElements);
   
    m_residual = ( m_im1.array() - m_im2Final.array() );
    m_rsquared = m_residual.array() * m_residual.array();

    m_weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
    m_weights = ( ( (m_residual.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH ).select( EdgeVO::Settings::HUBER_THRESH / (m_residual.array()).abs() , m_weights);

    if(flagGradMax){
        // for Geometric Errors
        m_residual_D.resize(numElements);
        m_rsquared_D.resize(numElements);    
        m_weights_D.resize(numElements);

        m_residual_D = (m_ZFinal.array() - m_D2Final.array());
        m_rsquared_D = m_residual_D.array() * m_residual_D.array();

        m_weights_D = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
        m_weights_D = ( ( (m_residual_D.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH ).select( EdgeVO::Settings::HUBER_THRESH / (m_residual_D.array()).abs() , m_weights);
    }

    return ( (m_weights.array() * m_rsquared.array()).sum() / (float) numElements );
     
}

/////////////////////////////////////////////////////////////////////////////////////
// TO DO LIST 2021.04.06.
// Iterative Weighted Least Square (IWLS) Problem for joint optimization.
// 0. Initialize Validity Maps based on RGB and Depth gradient values.
// 1. Photoconsistency Maximization
// 2. Gradient difference minimization
// 3. Point-to-Plane distance minimization 
//  - Change point-to-plane error based on the depth loss of Kerl13Iros.
//  - r_depth = D_2(tau(x,T)) - [(TK^(-1)(x,D_1(x)))]_Z
// TO DO LIST 2021.04.11.
/////////////////////////////////////////////////////////////////////////////////////
float EdgeDirectVO::warpAndProjectForAdaptiveDVO(const Eigen::Matrix<double,4,4>& invPose, int lvl, float& reweightDepthCost)
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif

    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;
    //std::cout << R << std::endl << t << std::endl;
    //std::cout << "Cols: " << m_X3D[lvl].cols() << "Rows: " << m_X3D[lvl].rows() << std::endl;
    
    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() ); // transform X3D of current frame to the reference camera coordinates.

    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //std::cout << cy << std::endl;
    //exit(1);
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

    m_warpedX.resize(m_X3D.rows());
    m_warpedY.resize(m_X3D.rows());
    m_warpedZ.resize(m_X3D.rows());

    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx; // x positon of the re-projected X3D on the reference frame.
    //m_warpedX.array() += cx;
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy; // y positon of the re-projected X3D on the reference frame.    
    //m_warpedY.array() += cy;
    m_warpedZ = m_newX3D.row(2).array(); // Z value of the transformed X3D on the reference frame.

    // (R.array() < s).select(P,Q );  // (R < s ? P : Q)
    //std::cout << newX3D.rows() << std::endl;
    //std::cout << m_finalMask.rows() << std::endl;

    // Check both Z 3D points are >0
    //m_finalMask = m_edgeMask;

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before Mask" << std::endl;
#endif

    // for Edge DVO
    m_finalMask = m_edgeMask;

    m_finalMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_newX3D.row(2).transpose().array() > EdgeVO::Settings::MAX_Z_DEPTH).select(0, m_finalMask);

    //m_finalMask = (m_newX3D.row(2).transpose().array() > 10.f).select(0, m_finalMask);
    m_finalMask = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_X3D.col(2).array() > 10.f).select(0, m_finalMask);
    m_finalMask = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMask, 0);
    m_finalMask = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMask, 0);
    
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMask = (m_warpedX.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array() >= w-2).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array().isFinite()).select(m_finalMask, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMask = (m_warpedY.array() >= h-2).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array().isFinite()).select(m_finalMask, 0);
    
    // for Adaptive DVO
    m_finalMaskGrad = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMaskGrad);
    //m_finalMask = (m_newX3D.row(2).transpose().array() > EdgeVO::Settings::MAX_Z_DEPTH).select(0, m_finalMask);

    //m_finalMask = (m_newX3D.row(2).transpose().array() > 10.f).select(0, m_finalMask);
    m_finalMaskGrad = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMaskGrad);
    //m_finalMask = (m_X3D.col(2).array() > 10.f).select(0, m_finalMask);
    m_finalMaskGrad = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMaskGrad, 0);
    m_finalMaskGrad = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMaskGrad, 0);
    
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMaskGrad = (m_warpedX.array() < 0.f).select(0, m_finalMaskGrad);
    m_finalMaskGrad = (m_warpedX.array() >= w-2).select(0, m_finalMaskGrad);
    m_finalMaskGrad = (m_warpedX.array().isFinite()).select(m_finalMaskGrad, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMaskGrad = (m_warpedY.array() >= h-2).select(0, m_finalMaskGrad);
    m_finalMaskGrad = (m_warpedY.array() < 0.f).select(0, m_finalMaskGrad);
    m_finalMaskGrad = (m_warpedY.array().isFinite()).select(m_finalMaskGrad, 0);
    
    // For non random numbers EDGEVO_SUBSET_POINTS
    size_t numElements = (m_finalMask.array() != 0).count();
    m_gxDFinal.resize(numElements);
    m_gyDFinal.resize(numElements);
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_D2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before pixel interpolation" << std::endl;
#endif

    size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {   
#if ADAPTIVE_DVO_FULL | ADAPTIVE_DVO_EDGE
            // gradient loss            
            if(m_finalMaskGrad[i] != 0){
                m_gxFinal[idx]  = interpolateVector( m_Gx, m_warpedX[i], m_warpedY[i], w);
                m_gyFinal[idx]  = interpolateVector( m_Gy, m_warpedX[i], m_warpedY[i], w);
                m_im1[idx] = m_grad1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
                m_im2Final[idx] = interpolateVector(m_grad2, m_warpedX[i], m_warpedY[i], w);
            }
            // photometric loss
            else{                
                m_gxFinal[idx]  = interpolateVector( m_gx, m_warpedX[i], m_warpedY[i], w);
                m_gyFinal[idx]  = interpolateVector( m_gy, m_warpedX[i], m_warpedY[i], w);
                m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
                m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
            }
#elif ADAPTIVE_DVO_WITHOUT_GRAD            
            // photometric loss            
            m_gxFinal[idx]  = interpolateVector( m_gx, m_warpedX[i], m_warpedY[i], w);
            m_gyFinal[idx]  = interpolateVector( m_gy, m_warpedX[i], m_warpedY[i], w);
            m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
            m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
#endif
            
            m_D2Final[idx]  = interpolateVector( m_D2, m_warpedX[i], m_warpedY[i], w);
            m_gxDFinal[idx]  = interpolateVector( m_gxD, m_warpedX[i], m_warpedY[i], w);
            m_gyDFinal[idx]  = interpolateVector( m_gyD, m_warpedX[i], m_warpedY[i], w);

            m_XFinal[idx] = m_newX3D(0,i);
            m_YFinal[idx] = m_newX3D(1,i);
            m_ZFinal[idx] = m_newX3D(2,i);

            ++idx;
        }
    }
    
    //apply mask to im1, im2, gx, and gy
    //interp coordinates of im2, gx, and gy
    // calc residual

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before Photometric Errors" << std::endl;
#endif

    //calc A and b matrices
    // for Photometric Errors
    m_residual.resize(numElements);
    m_rsquared.resize(numElements);
    m_weights.resize(numElements);
   
    m_residual = ( m_im1.array() - m_im2Final.array() );
    m_rsquared = m_residual.array() * m_residual.array();

    m_weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
    m_weights = ( ( (m_residual.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH ).select( EdgeVO::Settings::HUBER_THRESH / (m_residual.array()).abs() , m_weights);
    
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - before Geometric Errors" << std::endl;
#endif

    // for Geometric Errors
    m_residual_D.resize(numElements);
    m_rsquared_D.resize(numElements);    
    m_weights_D.resize(numElements);

    m_residual_D = (m_ZFinal.array() - m_D2Final.array());
    m_rsquared_D = m_residual_D.array() * m_residual_D.array();

    // std::cout << "Average residual Photo: " << m_residual.mean() << std::endl;
    // std::cout << "Average residual Geo: " << m_residual_D.mean() << std::endl;

    m_weights_D = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
    m_weights_D = ( ( (m_residual_D.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH_DEPTH ).select( EdgeVO::Settings::HUBER_THRESH_DEPTH / (m_residual_D.array()).abs() , m_weights_D);

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;
#endif

    // Compute weighted residuals.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> rsquared_W, rsqauredD_W;
    rsquared_W = m_weights.array() * m_rsquared.array();
    rsqauredD_W = m_weights_D.array() * m_rsquared_D.array();

#if ADAPTIVE_DVO_WITH_REWEIGHT_COST
    // Iteratively update lambda.
    // compute variance of each squared residual matrix.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> centered;
    float scale_p, scale_d;

    centered = rsquared_W.array() - rsquared_W.mean();    scale_p = sqrt((centered.array() * centered.array()).sum() / float(numElements - 1));
    centered = rsqauredD_W.array() - rsqauredD_W.mean();  scale_d = sqrt((centered.array() * centered.array()).sum() / float(numElements - 1));

    std::cout << "Variance of squared residual photometric: " << scale_p << std::endl;
    std::cout << "Variance of squared residual geometric: " << scale_d << std::endl;
    
    reweightDepthCost = scale_p/scale_d;

    std::cout << "Lambda: " << reweightDepthCost << std::endl;
#else
    // Fixed linear aggregation.
    reweightDepthCost = EdgeVO::Settings::COST_RATIO;
#endif

    return ( (rsquared_W + reweightDepthCost * rsqauredD_W).sum()/ (float) numElements );     
}

float EdgeDirectVO::computeAverageDisparity(const Eigen::Matrix<double,4,4>& invPose, int lvl)
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;
#endif

    float avg_disp = 0.0f;

    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> originX, originY;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> deltaX, deltaY, flowMag;

    //std::cout << R << std::endl << t << std::endl;
    //std::cout << "Cols: " << m_X3D[lvl].cols() << "Rows: " << m_X3D[lvl].rows() << std::endl;
    
    Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> m_X3Dt;

    m_X3Dt = m_X3D.transpose();

    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() ); // transform X3D of current frame to the reference camera coordinates.

    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //std::cout << cy << std::endl;
    //exit(1);
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

    // compute origin pixels.
    originX.resize(m_X3D.rows());
    originY.resize(m_X3D.rows());
    //m_originZ.resize(m_X3D.rows());

    originX = (fx * (m_X3Dt.row(0)).array() / (m_X3Dt.row(2)).array() ) + cx; // x positon of the re-projected X3D on the target frame.
    originY = (fy * (m_X3Dt.row(1)).array() / (m_X3Dt.row(2)).array() ) + cy; // y positon of the re-projected X3D on the target frame.    
    //m_originZ = m_X3D.row(2).array(); // Z value of the transformed X3D on the reference frame.

    // compute warped pixels.
    m_warpedX.resize(m_X3D.rows());
    m_warpedY.resize(m_X3D.rows());
    //m_warpedZ.resize(m_X3D.rows());

    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx; // x positon of the re-projected X3D on the reference frame.
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy; // y positon of the re-projected X3D on the reference frame.    
    //m_warpedZ = m_newX3D.row(2).array(); // Z value of the transformed X3D on the reference frame.

    // compute flows.
    deltaX.resize(m_X3D.rows());
    deltaY.resize(m_X3D.rows());
    flowMag.resize(m_X3D.rows());

    // std::cout << "m_X3D : " << m_X3D.rows() << "x" << m_X3D.cols() << std::endl;
    // std::cout << "m_newX3D : " << m_newX3D.rows() << "x" << m_newX3D.cols() << std::endl;
    // std::cout << "m_warpedX : " << m_warpedX.rows() << " x " << m_warpedX.cols() << std::endl;
    // std::cout << "originX : " << originX.rows() << " x " << originX.cols() << std::endl;

    deltaX = m_warpedX - originX;
    deltaY = m_warpedY - originY;
    flowMag = deltaX.array()*deltaX.array() + deltaY.array()*deltaY.array();

    avg_disp = cv::sqrt(flowMag.mean());
    
    return avg_disp;
}

void EdgeDirectVO::solveSystemOfEquations(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate)
{
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);

    size_t numElements = m_im2Final.rows();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

    m_Jacobian.resize(numElements, Eigen::NoChange);
    m_Jacobian.col(0) =  m_weights.array() * fx * ( m_gxFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(1) =  m_weights.array() * fy * ( m_gyFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(2) = - m_weights.array()* ( fx * ( m_XFinal.array() * m_gxFinal.array() ) + fy * ( m_YFinal.array() * m_gyFinal.array() ) )
                        / ( Z2.array() );

    m_Jacobian.col(3) = - m_weights.array() * ( fx * m_XFinal.array() * m_YFinal.array() * m_gxFinal.array() / Z2.array()
                         + fy *( 1.f + ( m_YFinal.array() * m_YFinal.array() / Z2.array() ) ) * m_gyFinal.array() );

    m_Jacobian.col(4) = m_weights.array() * ( fx * (1.f + ( m_XFinal.array() * m_XFinal.array() / Z2.array() ) ) * m_gxFinal.array() 
                        + fy * ( m_XFinal.array() * m_YFinal.array() * m_gyFinal.array() ) / Z2.array() );

    m_Jacobian.col(5) = m_weights.array() * ( -fx * ( m_YFinal.array() * m_gxFinal.array() ) + fy * ( m_XFinal.array() * m_gyFinal.array() ) )
                        / m_ZFinal.array();
    
    m_residual.array() *= m_weights.array();
    
    poseupdate = -( (m_Jacobian.transpose() * m_Jacobian).cast<double>() ).ldlt().solve( (m_Jacobian.transpose() * m_residual).cast<double>() );

    
}

/////////////////////////////////////////////////////////////////
void EdgeDirectVO::solveSystemOfEquationsForADVO(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate, float& reweightDepthCost)
{
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);

    size_t numElements = m_im2Final.rows();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

    // for Photometric Errors
    m_Jacobian.resize(numElements, Eigen::NoChange);
    m_Jacobian.col(0) =  m_weights.array() * ( fx * m_gxFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(1) =  m_weights.array() * ( fy * m_gyFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(2) = m_weights.array() * ( - ( fx * m_gxFinal.array() * m_XFinal.array() + fy * m_gyFinal.array() * m_YFinal.array() ) / Z2.array() );

    // m_Jacobian.col(3) = - m_ZFinal.array() * m_Jacobian.col(1).array() + m_YFinal.array() * m_Jacobian.col(2).array();

    // m_Jacobian.col(4) = m_ZFinal.array() * m_Jacobian.col(0).array() - m_XFinal.array() * m_Jacobian.col(2).array();

    // m_Jacobian.col(5) = -m_YFinal.array() * m_Jacobian.col(0).array() + m_YFinal.array() * m_Jacobian.col(1).array();

    m_Jacobian.col(3) = - m_weights.array() * ( fx * m_XFinal.array() * m_YFinal.array() * m_gxFinal.array() / Z2.array()
                         + fy *( 1.f + ( m_YFinal.array() * m_YFinal.array() / Z2.array() ) ) * m_gyFinal.array() );

    m_Jacobian.col(4) = m_weights.array() * ( fx * (1.f + ( m_XFinal.array() * m_XFinal.array() / Z2.array() ) ) * m_gxFinal.array() 
                        + fy * ( m_XFinal.array() * m_YFinal.array() * m_gyFinal.array() ) / Z2.array() );

    m_Jacobian.col(5) = m_weights.array() * ( -fx * ( m_YFinal.array() * m_gxFinal.array() ) + fy * ( m_XFinal.array() * m_gyFinal.array() ) )
                        / m_ZFinal.array();
   
    m_residual.array() *= m_weights.array();

    ////////////////////////////////////////////////
    // for Geometric Errors
    // for Z_2(tau(x,T))) - [Tpi^-1(x,Z_1(x))]_Z
    ////////////////////////////////////////////////
    m_Jacobian_D.resize(numElements, Eigen::NoChange);
    
    m_Jacobian_D.col(0) =  m_weights_D.array() * ( fx * m_gxDFinal.array() / m_ZFinal.array() );

    m_Jacobian_D.col(1) =  m_weights_D.array() * ( fy * m_gyDFinal.array() / m_ZFinal.array() );

    m_Jacobian_D.col(2) = m_weights_D.array() * ( - ( fx * m_gxDFinal.array() * m_XFinal.array() + fy * m_gyDFinal.array() * m_YFinal.array() ) / Z2.array() );
    
    m_Jacobian_D.col(3) = - m_ZFinal.array() * m_Jacobian_D.col(1).array() + m_YFinal.array() * m_Jacobian_D.col(2).array()
                            - m_weights_D.array();

    m_Jacobian_D.col(4) = m_ZFinal.array() * m_Jacobian_D.col(0).array() - m_XFinal.array() * m_Jacobian_D.col(2).array()
                            - m_weights_D.array() * m_YFinal.array();

    m_Jacobian_D.col(5) = -m_YFinal.array() * m_Jacobian_D.col(0).array() + m_YFinal.array() * m_Jacobian_D.col(1).array()
                            + m_weights_D.array() * m_XFinal.array();
    
    m_residual_D.array() *= m_weights_D.array();

    ///////////////////////////////////////////////////////
    // Tested empirically
    double ratio = reweightDepthCost, ratio_sq = sqrt(ratio);
    ///////////////////////////////////////////////////////

    poseupdate = -( (m_Jacobian.transpose() * m_Jacobian).cast<double>() + ratio * (m_Jacobian_D.transpose() * m_Jacobian).cast<double>()).ldlt().solve( 
        (m_Jacobian.transpose() * m_residual + ratio_sq * m_Jacobian_D.transpose() * m_residual_D).cast<double>() );
    // poseupdate = -( (m_Jacobian.transpose() * m_Jacobian).cast<double>() + reweightDepthCost * (m_Jacobian_D.transpose() * m_Jacobian).cast<double>()).ldlt().solve( 
    //     (m_Jacobian.transpose() * m_residual + reweightDepthCost * m_Jacobian_D.transpose() * m_residual_D).cast<double>() );    
}

// /////////////////////////////////////////////////////////////
// Under Construction 2021.09.19.
////////////////////////////////////////////////////////////////
// void EdgeDirectVO::solveSystemOfEquationsForADVO(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate, float& reweightDepthCost)
// {
//     const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
//     const float fx = cameraMatrix.at<float>(0, 0);
//     const float cx = cameraMatrix.at<float>(0, 2);
//     const float fy = cameraMatrix.at<float>(1, 1);
//     const float cy = cameraMatrix.at<float>(1, 2);

//     size_t numElements = m_im2Final.rows();
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

//     Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> m_Jacobian_W;
//     Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> m_JacobianD_W;

//     // for Photometric Errors
//     m_Jacobian.resize(numElements, Eigen::NoChange);
//     m_Jacobian.col(0) =  fx * m_gxFinal.array() / m_ZFinal.array();

//     m_Jacobian.col(1) =  fy * m_gyFinal.array() / m_ZFinal.array();

//     m_Jacobian.col(2) = - ( fx * m_gxFinal.array() * m_XFinal.array() + fy * m_gyFinal.array() * m_YFinal.array() ) / ( Z2.array() );

//     m_Jacobian.col(3) = - ( fx * m_gxFinal.array() * m_XFinal.array() * m_YFinal.array() / Z2.array() 
//                                 + fy * m_gyFinal.array() * ( 1.f + SQUARE(m_YFinal.array()) ) / Z2.array() );

//     m_Jacobian.col(4) = fx * m_gxFinal.array() * (1.f + SQUARE(m_XFinal.array()) / Z2.array() ) 
//                             + fy * m_gyFinal.array() * m_XFinal.array() * m_YFinal.array() / Z2.array();

//     m_Jacobian.col(5) = ( - fx * m_gxFinal.array() * m_YFinal.array() + fy * m_gyFinal.array() * m_XFinal.array() ) / m_ZFinal.array();

//     // multiply weights
//     m_Jacobian_W.resize(numElements, Eigen::NoChange);
//     for(int i=0; i<6; ++i) m_Jacobian_W.col(i) = m_weights.array() * m_Jacobian.col(i).array();    
//     //m_residual.array() *= m_weights.array();

//     ////////////////////////////////////////////////////////////
//     // for Geometric Errors
//     // for Z_2(tau(x,T))) - [Tpi^-1(x,Z_1(x))]_Z
//     ////////////////////////////////////////////////////////////
//     m_Jacobian_D.resize(numElements, Eigen::NoChange);    
    
//     m_Jacobian_D.col(0) =  fx * m_gxDFinal.array() / m_ZFinal.array();

//     m_Jacobian_D.col(1) =  fy * m_gyDFinal.array() / m_ZFinal.array();

//     m_Jacobian_D.col(2) = - ( fx * m_gxDFinal.array() * m_XFinal.array() + fy * m_gyDFinal.array() * m_YFinal.array() ) / ( Z2.array() );

//     m_Jacobian_D.col(3) = - ( fx * m_gxDFinal.array() * m_XFinal.array() * m_YFinal.array() / Z2.array() 
//                                 + fy * m_gyDFinal.array() * ( 1.f + SQUARE(m_YFinal.array()) ) / Z2.array() )
//                                 - 1.f;
//     // m_Jacobian_D.col(3) = - ( ( fx * m_XFinal.array() * m_YFinal.array() * m_gxDFinal.array() / Z2.array()
//     //                      + fy *( 1.f + ( m_YFinal.array() * m_YFinal.array() / Z2.array() ) ) * m_gyDFinal.array() )
//     //                      + 1.0);
    
//     m_Jacobian_D.col(4) =  fx * m_gxDFinal.array() * (1.f + SQUARE(m_XFinal.array()) / Z2.array() ) 
//                             + fy * m_gyDFinal.array() * m_XFinal.array() * m_YFinal.array() / Z2.array()
//                             - m_YFinal.array()
//                             + m_XFinal.array();
                        
//     // m_Jacobian_D.col(4) = ( ( fx * (1.f + ( m_XFinal.array() * m_XFinal.array() / Z2.array() ) ) * m_gxDFinal.array() 
//     //                     + fy * ( m_XFinal.array() * m_YFinal.array() * m_gyDFinal.array() ) / Z2.array() )
//     //                     + m_YFinal.array()
//     //                     - m_XFinal.array());

//     m_Jacobian_D.col(5) = ( - fx * m_gxDFinal.array() * m_YFinal.array() + fy * m_gyDFinal.array() * m_XFinal.array() ) / m_ZFinal.array();

//     // multiply weights
//     m_JacobianD_W.resize(numElements, Eigen::NoChange);
//     for(int i=0; i<6; ++i) m_JacobianD_W.col(i) = m_weights_D.array() * m_Jacobian_D.col(i).array();
//     //m_residual_D.array() *= m_weights_D.array();

//     ///////////////////////////////////////////////////////
//     // Tested empirically
//     double ratio = reweightDepthCost, ratio_sq = sqrt(ratio);
//     ///////////////////////////////////////////////////////

//     poseupdate = -( (m_Jacobian_W.transpose() * m_Jacobian).cast<double>() + ratio * (m_JacobianD_W.transpose() * m_Jacobian_D).cast<double>()).ldlt().solve( 
//         (m_Jacobian_W.transpose() * m_residual + ratio_sq * m_JacobianD_W.transpose() * m_residual_D).cast<double>() );

//     // poseupdate = -( (m_Jacobian_W.transpose() * m_Jacobian).cast<double>() + reweightDepthCost * (m_JacobianD_W.transpose() * m_Jacobian_D).cast<double>()).ldlt().solve( 
//     //       (m_Jacobian_W.transpose() * m_residual + reweightDepthCost * m_JacobianD_W.transpose() * m_residual_D).cast<double>() );
// }

float EdgeDirectVO::interpolateVector(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>& toInterp, float x, float y, int w) const
{
    int xi = (int) x;
	int yi = (int) y;
	float dx = x - xi;
	float dy = y - yi;
	float dxdy = dx * dy;
    int topLeft = w * yi + xi;
    int topRight = topLeft + 1;
    int bottomLeft = topLeft + w;
    int bottomRight= bottomLeft + 1;
  
    //               x                x+1
    //       ======================================
    //  y    |    topLeft      |    topRight      |
    //       ======================================
    //  y+w  |    bottomLeft   |    bottomRight   |
    //       ======================================
    return  dxdy * toInterp[bottomRight]
	        + (dy - dxdy) * toInterp[bottomLeft]
	        + (dx - dxdy) * toInterp[topRight]
			+ (1.f - dx - dy + dxdy) * toInterp[topLeft];
}
void EdgeDirectVO::prepare3DPoints( )
{
#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - X" << std::endl;    
#endif
    
    for (int lvl = 0; lvl < EdgeVO::Settings::PYRAMID_DEPTH; ++lvl)
    {
        const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
        int w = m_sequence.getFrameWidth(lvl);
        int h = m_sequence.getFrameHeight(lvl);
        const float fx = cameraMatrix.at<float>(0, 0);
        const float cx = cameraMatrix.at<float>(0, 2);
        const float fy = cameraMatrix.at<float>(1, 1);
        const float cy = cameraMatrix.at<float>(1, 2);
        const float fxInv = 1.f / fx;
        const float fyInv = 1.f / fy;
    
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                int idx = y * w + x;
                m_X3DVector[lvl].row(idx) << (x - cx) * fxInv, (y - cy) * fyInv, 1.f ;
            }
        }
    }

#ifdef DISPLAY_LOGS
    std::cout << typeid(*this).name() << "::" << __FUNCTION__ << " - E" << std::endl;    
#endif
}

void EdgeDirectVO::warpAndCalculateResiduals(const Pose& pose, const std::vector<float>& Z, const std::vector<bool>& E, const int h, const int w, const cv::Mat& cameraMatrix, const int lvl)
{
    const int ymax = h;
    const int xmax = w;
    const int length = xmax * ymax;

    const float fx = cameraMatrix.at<float>(0,0);
    const float cx = cameraMatrix.at<float>(0,2);
    const float fy = cameraMatrix.at<float>(1,1);
    const float cy = cameraMatrix.at<float>(1,2);

    const Mat inPose( m_trajectory.getCurrentPose().inversePose() );
    Eigen::Matrix<float,4,4> invPose;
    cv::cv2eigen(inPose,invPose);


    for(int i = 0; i < ymax*xmax; ++i)
    {
        float z3d = Z[i];
        float x = i / ymax;
        float y = i % xmax;
        float x3d = z3d * (x - cx)/ fx;
        float y3d = z3d * (y - cy)/ fy;
    }
    return;
}

inline
bool EdgeDirectVO::checkBounds(float x, float xlim, float y, float ylim, float oldZ, float newZ, bool edgePixel)
{
    return ( (edgePixel) & (x >= 0) & x < xlim & y >= 0 & y < ylim & oldZ >= 0. & newZ >= 0. );
        
}
void EdgeDirectVO::terminationRequested()
{
    printf("Display Terminated by User\n");
    m_statistics.printStatistics();

}

void EdgeDirectVO::outputPose(const Pose& pose, double timestamp)
{
    Eigen::Matrix<double,4,4,Eigen::RowMajor> T;
    cv::Mat pmat = pose.getPoseMatrix();
    cv::cv2eigen(pmat,T);
    Eigen::Matrix<double,3,3,Eigen::RowMajor> R = T.block<3,3>(0,0);
    Eigen::Matrix<double,3,Eigen::RowMajor> t = T.block<3,1>(0,3);
    Eigen::Quaternion<double> quat(R);

    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << timestamp;
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[0];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[1];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[2];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.x();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.y();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.z();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.w();
    m_outputFile << std::endl;
}



} //end namespace EdgeVO