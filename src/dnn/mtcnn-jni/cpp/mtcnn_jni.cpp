#include "mtcnn_jni.h"
#include "mtcnn/detector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

JNIEXPORT jobjectArray JNICALL Java_com_mtcnn_MTCNNDetector_detectFaces
  (JNIEnv *env, jobject obj, jstring modelPath, jstring imagePath, jfloat minFaceSize, jfloat scaleFactor) {
    
    // Chuyển đổi các tham số Java sang C++
    const char *modelPathStr = env->GetStringUTFChars(modelPath, 0);
    const char *imagePathStr = env->GetStringUTFChars(imagePath, 0);
    
    // Thiết lập cấu hình MTCNN
    std::string modelDir(modelPathStr);
    
    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = modelDir + "/det1.caffemodel";
    pConfig.protoText = modelDir + "/det1.prototxt";
    pConfig.threshold = 0.6f;
    
    RefineNetwork::Config rConfig;
    rConfig.caffeModel = modelDir + "/det2.caffemodel";
    rConfig.protoText = modelDir + "/det2.prototxt";
    rConfig.threshold = 0.7f;
    
    OutputNetwork::Config oConfig;
    oConfig.caffeModel = modelDir + "/det3.caffemodel";
    oConfig.protoText = modelDir + "/det3.prototxt";
    oConfig.threshold = 0.7f;
    
    // Khởi tạo detector và đọc ảnh
    MTCNNDetector detector(pConfig, rConfig, oConfig);
    cv::Mat img = cv::imread(imagePathStr);
    
    // Phát hiện khuôn mặt
    std::vector<Face> faces = detector.detect(img, minFaceSize, scaleFactor);
    
    // Giải phóng tài nguyên
    env->ReleaseStringUTFChars(modelPath, modelPathStr);
    env->ReleaseStringUTFChars(imagePath, imagePathStr);
    
    // Tìm lớp Face trong Java
    jclass faceClass = env->FindClass("com/mtcnn/Face");
    if (faceClass == NULL) {
        return NULL;
    }
    
    // Tìm constructor của lớp Face
    jmethodID constructor = env->GetMethodID(faceClass, "<init>", "(FFFFF[F)V");
    if (constructor == NULL) {
        return NULL;
    }
    
    // Tạo mảng đối tượng Face
    jobjectArray result = env->NewObjectArray(faces.size(), faceClass, NULL);
    
    // Điền dữ liệu vào mảng
    for (size_t i = 0; i < faces.size(); i++) {
        // Tạo mảng float cho landmarks
        jfloatArray landmarks = env->NewFloatArray(2 * NUM_PTS);
        env->SetFloatArrayRegion(landmarks, 0, 2 * NUM_PTS, faces[i].ptsCoords);
        
        // Tạo đối tượng Face
        jobject faceObj = env->NewObject(faceClass, constructor,
                                        faces[i].bbox.x1, faces[i].bbox.y1,
                                        faces[i].bbox.x2, faces[i].bbox.y2,
                                        faces[i].score, landmarks);
        
        // Thêm vào mảng kết quả
        env->SetObjectArrayElement(result, i, faceObj);
        
        // Giải phóng tài nguyên
        env->DeleteLocalRef(faceObj);
        env->DeleteLocalRef(landmarks);
    }
    
    return result;
}
