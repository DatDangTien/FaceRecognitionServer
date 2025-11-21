#pragma once

#include <pqxx/pqxx>
#include "utils.hpp"

struct DBPerson {
    std::string id;  // UUID as string
    std::string name;
    float confidence;
    float distance;
    
    DBPerson(
        const std::string& id,
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
        
        void insert_embedding(const std::string& name, const std::vector<float>& embedding, const std::string& robot = "");

        void update_embedding(const std::string& id, const std::vector<float>& embedding, const std::string& robot = "");
        
        std::vector<float> get_embedding(const std::string& name, const std::string& robot = "");
        
        std::vector<DBPerson> get_persons(std::string name, const std::string& robot = "");
        
        DBPerson get_recognition(const std::vector<float>& embedding, float threshold = 0.5);
};