#ifndef _MTCNN_JNI_H_
#define _MTCNN_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jobjectArray JNICALL Java_com_mtcnn_MTCNNDetector_detectFaces
  (JNIEnv *, jobject, jstring, jstring, jfloat, jfloat);

#ifdef __cplusplus
}
#endif

#endif // _MTCNN_JNI_H_
