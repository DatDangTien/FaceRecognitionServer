#pragma once

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>

class SubNet {
    public:
        SubNet();
        SubNet(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& model_path);
        ~SubNet();
        std::vector<Ort::Value> forward(Ort::Value& input_tensor);
        bool isInitialized() const { return initialized; }

    private:
        Ort::Session* session;
        bool initialized;
        size_t num_input;
        size_t num_output;
        std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
        std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
};