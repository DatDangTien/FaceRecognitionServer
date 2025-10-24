#pragma once

#include <pqxx/pqxx>
#include "utils.hpp"

struct DBPerson {
    uint64_t id;
    std::string name;
    float confidence;
    float distance;
    
    DBPerson(
        const uint64_t& id,
        const std::string& name,
        const float& confidence,
        const float& distance
    );
    
    std::string to_string() const;
};

class Postgres {
    private:
        pqxx::connection conn;
        
    public:
        Postgres(
            const std::string& host,
            const int& port,
            const std::string& dbname,
            const std::string& user,
            const std::string& password
        );
        
        ~Postgres();
        
        void insert_embedding(const std::string& name, const std::vector<float>& embedding);
        
        void update_embedding(const uint64_t& id, const std::string& name, const std::vector<float>& embedding);
        
        std::vector<float> get_embedding(const std::string& name);
        
        std::vector<DBPerson> get_persons(std::string name);
        
        DBPerson get_recognition(const std::vector<float>& embedding, float threshold = 0.5);
};