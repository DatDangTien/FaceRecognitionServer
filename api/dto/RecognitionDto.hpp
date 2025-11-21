#ifndef RECOGNITION_DTO_HPP
#define RECOGNITION_DTO_HPP

#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/core/Types.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

class RecognitionResultDto : public oatpp::DTO {
    DTO_INIT(RecognitionResultDto, DTO)
    
    DTO_FIELD(String, name);
    DTO_FIELD(Float32, confidence);
    DTO_FIELD(String, status);
    DTO_FIELD(Float32, xmin);
    DTO_FIELD(Float32, ymin);
    DTO_FIELD(Float32, xmax);
    DTO_FIELD(Float32, ymax);
    DTO_FIELD(String, person_id);
};

class RecognizeRequestDto : public oatpp::DTO {
    DTO_INIT(RecognizeRequestDto, DTO)
    
    DTO_FIELD(String, base64_image, "base64_image");
};

class RecognizeResponseDto : public oatpp::DTO {
    DTO_INIT(RecognizeResponseDto, DTO)
    
    DTO_FIELD(List<Object<RecognitionResultDto>>, faces);
    DTO_FIELD(Int32, total_faces);
};

class RegisterRequestDto : public oatpp::DTO {
    DTO_INIT(RegisterRequestDto, DTO)
    
    DTO_FIELD(String, name);  // Name as string
    DTO_FIELD(String, base64_image, "base64_image");
};

class RegisterResponseDto : public oatpp::DTO {
    DTO_INIT(RegisterResponseDto, DTO)
    
    DTO_FIELD(Boolean, success);
    DTO_FIELD(String, message);
};

class RootResponseDto : public oatpp::DTO {
    DTO_INIT(RootResponseDto, DTO)
    
    DTO_FIELD(String, message);
    DTO_FIELD(String, version);
    DTO_FIELD(Fields<String>, endpoints);
};

class HealthResponseDto : public oatpp::DTO {
    DTO_INIT(HealthResponseDto, DTO)
    
    DTO_FIELD(String, status);
    DTO_FIELD(Boolean, recognizer_initialized);
};

#include OATPP_CODEGEN_END(DTO)

#endif // RECOGNITION_DTO_HPP
