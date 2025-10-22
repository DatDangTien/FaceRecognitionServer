#include <pqxx/pqxx>
#include "utils.hpp"

struct PostgresPerson {
    uint64_t id;
    std::string name;
    float confidence;
    float distance;
    PostgresPerson(
        const uint64_t& id,
        const std::string& name,
        const float& confidence,
        const float& distance
    )
        : id(id), name(name), confidence(confidence), distance(distance) {
    }
    std::string to_string() const {
        return "ID: " + std::to_string(id) + ", Name: " + name + ", Confidence: " + std::to_string(confidence) + ", Distance: " + std::to_string(distance);
    }
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
            const std::string& password) 
            : conn("host=" + host + " port=" + std::to_string(port) + " dbname=" + dbname + " user=" + user + " password=" + password) {
        }
        ~Postgres() {
            conn.disconnect();
        }
        void insert_embedding(const std::string& name, const std::vector<float>& embedding) {
            pqxx::work txn(conn);
            txn.exec_params(
                "INSERT INTO Person(name, embedding) VALUES ($1, $2)",
                name,
                vec2pgvector(embedding)
            );
            txn.commit();
        }
        void update_embedding(const uint64_t& id, const std::string& name, const std::vector<float>& embedding) {
            pqxx::work txn(conn);
            txn.exec_params(
                "UPDATE Person SET name = $1, embedding = $2 WHERE id = $3",
                name,
                vec2pgvector(embedding),
                id
            );
            txn.commit();
        }
        std::vector<float> get_embedding(const std::string& name) {
            pqxx::work txn(conn);
            pqxx::result r = txn.exec_params(
                "SELECT embedding FROM Person WHERE name = $1",
                name
            );
            txn.commit();
            return pgvector2vec(r[0]["embedding"].as<std::string>());
        }
        PostgresPerson get_recognition(const std::vector<float>& embedding, float threshold = 0.5) {
            pqxx::work txn(conn);
            pqxx::result r = txn.exec_params(
                "SELECT id, name, distance "
                "FROM ( "
                "   SELECT id, name, embedding <=> $1::vector AS distance "
                "   FROM Person "
                ") AS sub "
                "WHERE distance < $2 "
                "ORDER BY distance "
                "LIMIT 1;",
                vec2pgvector(embedding),
                threshold
            );
            txn.commit();
            if (r.empty()) {
                return PostgresPerson(0, "", 0.0, 0.0);
            }
            return PostgresPerson(
                r[0]["id"].as<uint64_t>(),
                r[0]["name"].as<std::string>(),
                1 - r[0]["distance"].as<float>(),
                r[0]["distance"].as<float>()
            );
        };
    };