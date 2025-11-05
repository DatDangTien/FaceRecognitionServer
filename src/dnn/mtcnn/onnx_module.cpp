#include "onnx_module.h"

void get_node_names(
    Ort::Session& session,
    bool is_input,
    std::vector<Ort::AllocatedStringPtr>& names_ptrs,
    std::vector<const char*>& names
) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_nodes = is_input ? session.GetInputCount() : session.GetOutputCount();
    
    for (size_t i = 0; i < num_nodes; i++) {
        auto name_ptr = is_input ? 
            session.GetInputNameAllocated(i, allocator) :
            session.GetOutputNameAllocated(i, allocator);
        names_ptrs.push_back(std::move(name_ptr));
        names.push_back(names_ptrs.back().get());
    }
}

SubNet::SubNet() : session(nullptr), initialized(false), num_input(0), num_output(0) {}

SubNet::SubNet(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& model_path)
    : initialized(false) {
    try {
        session = new Ort::Session(env, model_path.c_str(), session_options);
        
        get_node_names(*session, true, input_names_ptrs, input_names);
        get_node_names(*session, false, output_names_ptrs, output_names);
        
        num_input = input_names.size();
        num_output = output_names.size();
        initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model " << model_path << ": " << e.what() << std::endl;
        session = nullptr;
        initialized = false;
    }
}

SubNet::~SubNet() {
    if (session) {
        delete session;
        session = nullptr;
    }
}

std::vector<Ort::Value> SubNet::forward(Ort::Value& input_tensor) {
    if (!initialized || !session) {
        throw std::runtime_error("SubNet not properly initialized");
    }
    
    return session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        num_input,
        output_names.data(),
        num_output
    );
}